import os
import json
import yaml
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Literal

import ast
import torch.nn.functional as F

from ...utils import logging as logging_utils
from .transform import Normalizer, prepare_images, prepare_state, prepare_language, prepare_action, expert_visual_transform, apply_visual_augmentation
from .wrist_augment import DEFAULT_WRIST_CAMERAS, augment_wrist_views
from .ee_pose_transform import *
from typing import Dict, List, Optional
import ast
import torch.nn.functional as F

logger = logging_utils.get_logger(__name__)

# Supported robot action/state spaces.
#   'joint' -> absolute joint positions (per-arm joint angles + gripper)
#   'ee'    -> absolute end-effector poses (per-arm xyz + quaternion + gripper)
# The space selects which robot config is loaded, which in turn decides how the
# raw lerobot columns are sliced into the unified `arm.position` /
# `end.position` / `effector.position` feature space.
ACTION_SPACES = ('joint', 'ee')

# Absolute vs relative action targets, an axis orthogonal to `action_space`:
#   'abs' -> the action column is the target pose itself
#   'rel' -> the action column is the target pose minus the current state
# In the robot config this is what `subtract_state` encodes per feature; the
# value here declares which of the two a run is, so the config and the
# representation can be checked against each other at startup.
ACTION_TYPES = ('abs', 'rel')

# Robot config filename suffix per action space. 'joint' keeps the historical
# `<robot>.yaml` name so existing configs and checkpoints are unaffected.
_ACTION_SPACE_SUFFIX = {'joint': '', 'ee': '_ee'}

# Robot config filename suffix per action type. 'abs' keeps the historical
# `<robot>.yaml` / `<robot>_ee.yaml` names, so pre-existing configs, norm stats
# and checkpoints keep resolving to exactly the same file.
_ACTION_TYPE_SUFFIX = {'abs': '', 'rel': '_rel'}

# Unmasked cam_high frames of a SAM dataset, loaded next to the masked view
# when `data.aux_raw_image` is on. They only feed the depth / video teachers
# (and the patch weights derived from the raw/masked pair); the policy still
# sees the masked image.
RAW_CAM_HIGH_KEY = 'observation.images.cam_high_raw'


def derive_keep_mask(raw, masked, outside_scale):
    """Recover the SAM keep-mask from a raw frame and its masked rendering.

    The masked frame is ``raw`` inside the mask and ``raw * outside_scale``
    outside (plus an optional white outline ring just outside the mask). Each
    pixel is assigned to whichever of the two renderings it is closer to; the
    outline ring is near-white while the raw pixel is not, and is excluded.
    Against the saved FFV1 masks this agrees on ~98.6-99.97% of pixels.

    Args: uint8 or float tensors (..., 3, H, W) on the same scale.
    Returns: float tensor (..., 1, H, W) in {0, 1}.
    """
    raw = raw.float()
    masked = masked.float()
    d_in = (masked - raw).abs().mean(dim=-3, keepdim=True)
    d_out = (masked - raw * outside_scale).abs().mean(dim=-3, keepdim=True)
    keep = d_in < d_out
    ring = (masked.amin(dim=-3, keepdim=True) >= 235) & (d_in > 40)
    return (keep & ~ring).float()


def normalize_action_space(action_space):
    """Validate an action space value, mapping None to the default 'joint'."""
    if action_space is None:
        return 'joint'
    if action_space not in ACTION_SPACES:
        raise ValueError(
            f"Unsupported action_space {action_space!r}; expected one of {list(ACTION_SPACES)}."
        )
    return action_space


def normalize_action_type(action_type):
    """Validate an action type value, mapping None to the default 'abs'.

    A missing `action_type` means absolute, which is what every config written
    before this field existed contains -- reading such a run must keep behaving
    exactly as it did, so None (and only None) is accepted as the default.
    """
    if action_type is None:
        return 'abs'
    if action_type not in ACTION_TYPES:
        raise ValueError(
            f"Unsupported action_type {action_type!r}; expected one of {list(ACTION_TYPES)}."
        )
    return action_type


def infer_action_type(action_subtract_state, actions):
    """Derive the action type a robot config actually implements.

    `subtract_state` is what really decides whether a target is a delta, so for a
    config that does not declare `action_type` this recovers the representation
    from it. Configs written before `action_type` existed (notably
    `configs/robot_configs/agilex_cobot_magic.yaml`, the documented real-robot
    example, and the RoboTwin configs trained at commit 7afe3fe) rely on this:
    they are read exactly as before rather than being forced into 'abs'.

    Only pose features count. Effector features are never subtractive, so a
    gripper-only or all-absolute config stays 'abs'.
    """
    pose_keys = [a for a in actions if a.endswith('arm.position') or a.endswith('end.position')]
    if any(action_subtract_state.get(a) for a in pose_keys):
        return 'rel'
    return 'abs'

def resolve_robot_config_path(robot_config_root, data_name, action_space='joint',
                              action_type='abs'):
    """Resolve the robot config path for `data_name`, space and type.

    The two axes compose into four config files:

        joint + abs -> <robot>.yaml          (historical)
        joint + rel -> <robot>_rel.yaml
        ee    + abs -> <robot>_ee.yaml
        ee    + rel -> <robot>_ee_rel.yaml

    Training and inference both go through this helper so a checkpoint can never
    be served with a robot config describing a different representation (which
    would silently reinterpret, say, EE poses as joint angles).
    """
    action_space = normalize_action_space(action_space)
    action_type = normalize_action_type(action_type)
    suffix = _ACTION_SPACE_SUFFIX[action_space] + _ACTION_TYPE_SUFFIX[action_type]
    path = os.path.join(robot_config_root, f'{data_name}{suffix}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Robot config {path} not found for data_name={data_name!r} with "
            f"action_space={action_space!r} and action_type={action_type!r}. "
            f"Each representation needs its own robot config: 'joint'+'abs' reads "
            f"<robot>.yaml, 'joint'+'rel' reads <robot>_rel.yaml, 'ee'+'abs' reads "
            f"<robot>_ee.yaml and 'ee'+'rel' reads <robot>_ee_rel.yaml."
        )
    return path


def compute_image_token_count(images, image_grid_thw=None, merge_size=2, use_vision_boundaries=True):
    if not isinstance(images, torch.Tensor) or images.numel() == 0:
        return 0

    boundary_tokens = 2 if use_vision_boundaries else 0
    if image_grid_thw is not None:
        grid = image_grid_thw
        if not isinstance(grid, torch.Tensor):
            grid = torch.as_tensor(grid)
        grid = grid.reshape(-1, 3)
        num_patch = grid.prod(dim=-1) // (int(merge_size) ** 2)
        return int((num_patch + boundary_tokens).sum().item())

    if images.ndim == 3:
        num_images, num_patch = images.shape[0], images.shape[1]
    elif images.ndim >= 4:
        num_images, num_patch = images.shape[0], images.shape[-2]
    else:
        return 0
    return int(num_images * (num_patch + boundary_tokens))


class FeatureInfo(BaseModel):
    joints: List[str] | None = None
    images: List[str] | None = None
    joints_max_dim: dict | None = None

    def update_info(self, data_config):
        joints_info = data_config.joints
        self.images = ['observation.images.'+image for image in data_config.cameras]

        joints= []
        joints_max_dim = {}
        for s in joints_info:
            joint_info = ast.literal_eval(s)
            joint = next(iter(joint_info.keys()))
            if joint_info[joint] == 0: continue

            joints.append(joint)
            joints_max_dim.update(joint_info)
        self.joints = joints
        self.joints_max_dim = joints_max_dim


class FeatureTransform:
    def __init__(
        self,
        robot_config_path,
        data_config,
        model_config,
        processor,
        disabled_image_features=False,
        do_nomalize=True,
        chunk_size=50,
        return_item_befor_padding=False,
        norm_stats_path=None,
        use_depth_align=False,
        image_augment=False,
        use_future_image=False,
        action_space='joint',
        action_type='abs',
        aux_raw_image=False,):

        assert os.path.exists(robot_config_path), f"{robot_config_path} does not exist."
        with open(robot_config_path, 'r') as f:
            robot_config = yaml.safe_load(f)
        f.close()

        self.action_space = normalize_action_space(action_space)
        # The declared type drives validation; the resolved type is what the run
        # actually is. They differ only when `action_type` was never declared, in
        # which case the robot config's own `subtract_state` decides -- see
        # `infer_action_type`. `action_type` (declared) stays raw so the checker
        # can tell "explicitly abs" from "unspecified".
        self.declared_action_type = action_type
        self.action_type = normalize_action_type(action_type)

        if norm_stats_path is None:
            norm_stats_path = robot_config.pop('norm_stats')
        else:
            robot_config.pop('norm_stats')
            if not os.path.exists(norm_stats_path):
                raise FileNotFoundError(
                    f"norm stats file {norm_stats_path!r} does not exist. This path "
                    f"came from the run config's `norm_stats_file` (training) or the "
                    f"checkpoint's lingbotvla_cli.yaml (deployment), which takes "
                    f"precedence over {robot_config_path}'s own `norm_stats:` key. "
                    f"Fix the path, or clear it to fall back to the robot config's "
                    f"`norm_stats:` ({robot_config.get('norm_stats')!r})."
                )

        # Which quaternion layout the DATASET stores for EE poses. Popped for the
        # same reason as `norm_stats`: check_robot_config and get_feature_mapping
        # iterate the remaining top-level keys and require every value to be a
        # list of feature dicts, so a scalar key left in place would break them.
        #
        # ee_pose_transform's helpers are XYZW; RoboTwin (and therefore the
        # RoboTwin lerobot EE datasets) is WXYZ. 'xyzw' stays the default so
        # existing configs are untouched, and apply/unapply convert at the
        # boundary when this says 'wxyz'.
        self.quat_convention_declared = 'quat_convention' in robot_config
        self.quat_convention = robot_config.pop('quat_convention', 'xyzw')
        if self.quat_convention not in QUAT_CONVENTIONS:
            raise ValueError(
                f"{robot_config_path} sets quat_convention="
                f"{self.quat_convention!r}; expected one of {list(QUAT_CONVENTIONS)}."
            )


        self.feature_config = FeatureInfo()
        if getattr(data_config, 'joints', None) is not None:
            self.feature_config.update_info(data_config)

        if not return_item_befor_padding: self.check_robot_config(robot_config)

        self.model_config = model_config
        self.tokenizer = processor.tokenizer if processor is not None else None
        self.processor = processor

        self.chunk_size = chunk_size
        self.return_item_befor_padding = return_item_befor_padding

        # disabled_image_features: keep the image features or not when getting lerobot item
        self.disabled_image_features = disabled_image_features
        self.use_depth_align = use_depth_align
        self.use_future_image = use_future_image
        self.aux_raw_image = bool(aux_raw_image) and use_depth_align
        # Raw/masked pair -> per-patch foreground weight for the teacher losses.
        # None -> filled from the dataset's sam_info.json by VLADataset.
        self.aux_outside_scale = getattr(data_config, 'aux_outside_scale', None)

        if not disabled_image_features:
            self.image_augment = image_augment

        # Training-only photometric augmentation of the wrist views (see wrist_augment.py).
        # Read from data_config so deploy, which rebuilds this from lingbotvla_cli.yaml,
        # sees the flag too; apply() skips it under policy_eval.
        self.wrist_augment = bool(getattr(data_config, 'wrist_augment', False)) and not disabled_image_features
        self.wrist_view_keys = []
        if self.wrist_augment:
            cameras = getattr(data_config, 'wrist_augment_cameras', None) or list(DEFAULT_WRIST_CAMERAS)
            self.wrist_view_keys = ['observation.images.' + c for c in cameras]
            missing = [k for k in self.wrist_view_keys if k not in (self.feature_config.images or [])]
            if missing:
                raise ValueError(
                    f"data.wrist_augment_cameras {cameras} not in data.cameras "
                    f"{getattr(data_config, 'cameras', None)}: {missing}"
                )
            logger.info_rank0(f"wrist_augment on (train only): {self.wrist_view_keys}")

        # keep the self.feature_to_keep in lerobot item when convert to new item
        self.feature_to_keep = set([
            'timestamp',
            'frame_index',
            'episode_index',
            'task_index',
            'action_is_pad',
            'task',
        ])

        target_features  = {'states':[], 'actions':[], 'images':[]}
        org_features  = {'states':set(), 'actions':set(), 'images':set()}
        self.get_feature_mapping(robot_config, target_features, org_features)
        self.states = target_features['states']
        self.actions = target_features['actions']
        self.images = target_features['images']

        self.org_features = org_features
        if self.aux_raw_image:
            self.feature_to_keep.add(RAW_CAM_HIGH_KEY)
            # The teachers read image slot 0 (see get_depth_target / get_video_target);
            # the raw frames replace exactly that slot, so it must be cam_high.
            first = self.images[0] if self.images else None
            origin = self.key_mapping.get(first, {}).get('origin_keys') if first else None
            if origin != 'observation.images.cam_high':
                raise ValueError(
                    f"aux_raw_image needs the first camera to map to observation.images.cam_high, "
                    f"got {first!r} <- {origin!r}."
                )

        self.check_action_space(robot_config_path)
        self.check_action_type(robot_config_path)

        self.normalizer = self.get_normalizer(norm_stats_path, do_nomalize, data_config)

    def check_action_space(self, robot_config_path):
        """Fail fast when the robot config does not match the declared action space.

        `action_space` only chooses which robot config file is read, so a wrong
        pairing would otherwise train happily on the wrong representation and
        only show up as a silently broken policy at eval time.
        """
        arm_keys = [a for a in self.actions if a.endswith('arm.position')]
        ee_keys = [a for a in self.actions if a.endswith('end.position')]

        if self.action_space == 'joint':
            if not arm_keys or ee_keys:
                raise ValueError(
                    f"action_space='joint' expects the robot config to drive "
                    f"action.arm.position and no action.end.position, but "
                    f"{robot_config_path} defines actions {self.actions}."
                )
        elif self.action_space == 'ee':
            if not ee_keys or arm_keys:
                raise ValueError(
                    f"action_space='ee' expects the robot config to drive "
                    f"action.end.position and no action.arm.position, but "
                    f"{robot_config_path} defines actions {self.actions}."
                )

    def check_action_type(self, robot_config_path):
        """Fail fast when the robot config does not match the declared action type.

        `action_type` is the abs/rel axis and `subtract_state` (per feature in the
        robot config) is how relative is actually implemented, so a mismatch would
        otherwise train a run whose checkpoints are labelled with a representation
        they do not use. Two consequences are checked here because both are silent
        rather than loud:

        - a *relative* run whose pose features are not subtractive would label
          absolute targets as deltas, and inference would add a state back onto a
          value that never had one removed;
        - a *relative* pose feature must not carry a quaternion `relative_type`,
          since that routes through rotation composition and asserts on the joint
          pose feature mid-dataloading instead of failing here.

        Only an *explicitly declared* `action_type` is validated. When it is
        omitted the representation is inferred from the config instead, so runs
        that predate the field (and configs whose abs/rel is expressed purely
        through `subtract_state`, such as the real-robot example) keep working
        untouched rather than being forced into 'abs'.

        The gripper is deliberately exempt from the 'rel' requirement: effector
        features cannot be subtractive (`get_feature_mapping` asserts this), so a
        relative run necessarily keeps it absolute.
        """
        pose_keys = [a for a in self.actions if a.endswith('arm.position') or a.endswith('end.position')]
        assert pose_keys, f"{robot_config_path} defines no pose action feature: {self.actions}"

        subtractive = [a for a in pose_keys if self.action_subtract_state.get(a)]

        # Resolve what this run actually is, and record it so downstream callers
        # (the deploy server, norm-stats computation) see one consistent answer.
        if self.declared_action_type is None:
            self.action_type = infer_action_type(self.action_subtract_state, self.actions)
            logger.info(
                "action_type not declared; inferred %r from %s (subtractive pose "
                "features: %s)", self.action_type, robot_config_path, sorted(subtractive) or 'none',
            )
            if self.action_type == 'rel' and _is_quaternion_relative_type(
                self.action_relative_type.get(pose_keys[0])
            ):
                # No declared type to contradict, but the quaternion/joint pairing is
                # never valid, so still surface it here instead of mid-dataloading.
                self._raise_quaternion_joint_error(robot_config_path, pose_keys)
            if self.action_type == 'rel':
                self._warn_undeclared_quat_convention(robot_config_path, pose_keys)
            return

        if self.action_type == 'rel':
            if len(subtractive) != len(pose_keys):
                raise ValueError(
                    f"action_type='rel' expects every pose action feature to set "
                    f"`subtract_state: True`, but {robot_config_path} has "
                    f"{sorted(set(pose_keys) - set(subtractive))} set to False. "
                    f"Note that a relative run needs its own config file "
                    f"(<robot>_rel.yaml) and its own norm stats."
                )
            self._raise_quaternion_joint_error(robot_config_path, pose_keys)
            self._warn_undeclared_quat_convention(robot_config_path, pose_keys)
        else:
            if subtractive:
                raise ValueError(
                    f"action_type='abs' expects every pose action feature to set "
                    f"`subtract_state: False`, but {robot_config_path} sets it True for "
                    f"{sorted(subtractive)}. Either point action_type at 'rel', or drop "
                    f"the field entirely to let it be inferred from the config."
                )

    def _warn_undeclared_quat_convention(self, robot_config_path, pose_keys):
        """Warn when a relative EE run leaves `quat_convention` at its default.

        The default is 'xyzw' for backwards compatibility, but RoboTwin data is
        WXYZ. Getting this wrong is silent: the relative/absolute round trip stays
        self-consistent, so nothing raises -- the model simply learns deltas
        computed from a wrong rotation (and, under `quaternion_local`, a wrong
        translation whose axes are permuted and sign flipped).
        """
        if self.quat_convention_declared:
            return  # explicitly declared, nothing to guess
        quat_pose_keys = [
            k for k in pose_keys
            if k.endswith('end.position')
            and _is_quaternion_relative_type(self.action_relative_type.get(k))
        ]
        if not quat_pose_keys:
            return
        logger.warning(
            "%s does not declare `quat_convention`, so the EE quaternions in %s "
            "are assumed to be %r. RoboTwin stores WXYZ (transforms3d.mat2quat); "
            "if this dataset came from RoboTwin, add `quat_convention: wxyz` to "
            "the robot config. A wrong value does not raise -- it silently trains "
            "on incorrect rotation deltas.",
            robot_config_path, sorted(quat_pose_keys), self.quat_convention,
        )

    def _raise_quaternion_joint_error(self, robot_config_path, pose_keys):
        """Reject a quaternion `relative_type` on a joint pose feature.

        The quaternion branch composes rotations for xyz+quaternion poses and
        asserts `end.position` is present, so on a joint feature it fails with a
        confusing assertion part-way through dataloading.
        """
        for feature in pose_keys:
            if feature.endswith('end.position'):
                continue  # quaternion types are the correct EE representation
            relative_type = self.action_relative_type.get(feature)
            if _is_quaternion_relative_type(relative_type):
                raise ValueError(
                    f"{robot_config_path} sets the joint pose feature {feature!r} to the "
                    f"quaternion relative_type {relative_type!r}, which routes through "
                    f"rotation composition for xyz+quaternion poses and asserts on "
                    f"`end.position` instead. Set an explicit plain `relative_type` "
                    f"(e.g. 'world') for joint delta actions."
                )

    def get_normalizer(self, norm_stats_path, do_nomalize, data_config):

        if not do_nomalize: return None

        action_state_norm_type = {k: v for d in data_config.norm_type for k, v in ast.literal_eval(d).items()} 

        assert norm_stats_path is not None
        norm_type = {}
        for feature in self.actions:
            base_name = feature.split('action.')[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]

        for feature in self.states:
            base_name = feature.split('observation.state.')[-1]
            assert base_name in action_state_norm_type, f"{feature} does not have predefined norm type."
            norm_type[feature] = action_state_norm_type[base_name]
        
        for feature in self.images:
            norm_type[feature] = 'identity'

        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        f.close()

        normalizer = Normalizer(
            norm_stats=norm_stats['norm_stats'],
            norm_type=norm_type,
        )

        return normalizer


    def check_robot_config(self, robot_config):

        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)
            for feature_convert_info in features_convert_info:

                if isinstance(feature_convert_info, dict):
                    assert len(feature_convert_info.keys()) == 1

                if feature_category == 'actions':
                    assert isinstance(feature_convert_info, dict)
                    action_feature = list(feature_convert_info.keys())[0].split('action.')[-1]
                    if action_feature not in self.feature_config.joints:
                        raise ValueError(f"{action_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}")

                elif feature_category == 'states':
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        state_feature = list(feature_convert_info.keys())[0]
                    else: 
                        state_feature = feature_convert_info
                    state_feature = state_feature.split('observation.state.')[-1]
                    if state_feature not in self.feature_config.joints:
                        raise ValueError(f"{state_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.joints}")

                elif feature_category == 'images':
                    assert isinstance(feature_convert_info, dict) or isinstance(feature_convert_info, str)
                    if isinstance(feature_convert_info, dict):
                        image_feature = list(feature_convert_info.keys())[0]
                    else: 
                        image_feature = feature_convert_info
                    if image_feature not in self.feature_config.images:
                        raise ValueError(f"{image_feature} in the robot config is not included among the predefined features in the training config: {self.feature_config.images}")


    def get_feature_mapping(self, robot_config, target_features, org_features):
        to_convert_features = {}
        reverse_convert_features = {}
        action_subtract_state = {}
        action_relative_type = {}
        actions_convert_from_state = set()
        for feature_category, features_convert_info in robot_config.items():
            assert isinstance(features_convert_info, list)

            for feature_convert_info in features_convert_info:
                
                if isinstance(feature_convert_info, str):
                    self.feature_to_keep.add(feature_convert_info)
                    target_features[feature_category].append(feature_convert_info)
                    org_features[feature_category].add(feature_convert_info)
                    
                elif isinstance(feature_convert_info, dict):
                    target_feature = next(iter(feature_convert_info.keys()))
                    target_features[feature_category].append(target_feature)

                    if feature_category == 'actions':
                        assert isinstance(feature_convert_info, dict)
                        assert 'subtract_state' in feature_convert_info[target_feature]
                        action_subtract_state[target_feature] = feature_convert_info[target_feature].pop('subtract_state')
                        if 'velocity' in target_feature or 'effector.position' in target_feature:
                            assert not action_subtract_state[target_feature], f"{target_feature} cannot be subtracted from state"

                        action_relative_type[target_feature] = feature_convert_info[target_feature].pop('relative_type', 'quaternion_local')
                        if_convert_from_state = feature_convert_info[target_feature].pop('convert_from_state', False)
                        if if_convert_from_state:
                            actions_convert_from_state.add(target_feature)

                        if 'origin_keys' not in feature_convert_info[target_feature] and not if_convert_from_state:
                            self.feature_to_keep.add(feature_convert_info)

                    if 'origin_keys' in feature_convert_info[target_feature]:
                        if isinstance(feature_convert_info[target_feature]['origin_keys'], list):
                            ordered_origin_keys = OrderedDict()
                            for item in feature_convert_info[target_feature]['origin_keys']:
                                for k, v in item.items():
                                    while k in ordered_origin_keys:
                                        k = k+'*'
                                    ordered_origin_keys[k] = v

                            feature_convert_info[target_feature]['origin_keys'] = ordered_origin_keys
                            to_convert_features.update(feature_convert_info)

                            if feature_category in ['actions', 'states']:
                                org_features[feature_category].update([k.split('*')[0] for k in ordered_origin_keys.keys()])

                            target_start_id = 0
                            for org_key, info in ordered_origin_keys.items():
                                org_info = info.copy()
                                org_key = org_key.split('*')[0]
                                if org_key not in reverse_convert_features:
                                    reverse_convert_features[org_key] = []

                                if 'start' in org_info:
                                    org_info['target_key'] = target_feature
                                    org_info['target_start'] = target_start_id
                                    org_info['target_end'] = target_start_id+org_info['end']-org_info['start']
                                    target_start_id =  org_info['target_end']
                                else:
                                    org_info['target_key'] = target_feature
                                reverse_convert_features[org_key].append(org_info)


                        if isinstance(feature_convert_info[target_feature]['origin_keys'], str):
                            to_convert_features[target_feature] = feature_convert_info[target_feature]
                            reverse_convert_features[feature_convert_info[target_feature]['origin_keys']] = {'target_key': target_feature}
                            org_features[feature_category].add(feature_convert_info[target_feature]['origin_keys'])

        to_convert_state = {}
        if len(actions_convert_from_state)>0:
            for action in list(actions_convert_from_state):

                assert action.replace("action.", "observation.state.") in target_features['states'], f"{action} can not converted from {action.replace('action.', 'observation.state.')}"
                to_convert_state[action] = action.replace("action.", "observation.state.")
        self.actions_convert_from_state = to_convert_state
        
        if len(self.actions_convert_from_state)>0:
            # To keep time alignment, if any action is converted from state, all actions must be converted from state
            assert len(self.actions_convert_from_state)==len(target_features['actions'])

        self.action_subtract_state = action_subtract_state
        self.action_relative_type = action_relative_type
        for feature_category, feature in org_features.items():
            if feature_category == 'actions':
                if len(feature) == 0 and len(self.actions_convert_from_state)>0:
                    org_features['actions'] == []
                    continue
            if len(feature) == 0:
                org_features[feature_category] = target_features[feature_category]
            else: org_features[feature_category] = list(feature)

        self.key_mapping = to_convert_features
        self.key_reverse_mapping = reverse_convert_features
        

    def convert_features(self, item, w_action):
        out_item = {}
        for target_key, convert_info in self.key_mapping.items():
            if self.disabled_image_features and target_key in self.images:
                continue
            if not w_action and 'action' in target_key:
                continue
            if isinstance(convert_info['origin_keys'], str) and convert_info['origin_keys'] in item:
                out_item[target_key] = item[convert_info['origin_keys']]
                continue

            assert isinstance(convert_info['origin_keys'], OrderedDict)
            concat_list = []
            convert_success = True
            for origin_key, origin_info in convert_info['origin_keys'].items():
                origin_key = origin_key.split('*')[0]
                if origin_key not in item:
                    convert_success = False
                    break
                origin_data = item.get(origin_key)[..., origin_info['start']:origin_info['end']]
                concat_list.append(origin_data)
            if convert_success:
                out_item[target_key] = torch.cat(concat_list, dim=-1)
            del concat_list

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]
        
        if len(self.actions_convert_from_state)>0 and w_action:
            for action_feature in self.actions:
                state_feature = self.actions_convert_from_state[action_feature]
                assert state_feature in out_item and len(out_item[state_feature].shape) == 2
                out_item[action_feature] = out_item[state_feature][1:].clone()
                out_item[state_feature] = out_item[state_feature][0].clone()
            for state_feature in self.states:
                if len(out_item[state_feature].shape) == 2:
                    out_item[state_feature] = out_item[state_feature][0]

        del item
        return out_item

    def reverse_features(self, item):
        if len(self.actions_convert_from_state)>0:
            for action_feature, state_feature in self.actions_convert_from_state.items():
                item[state_feature] = torch.cat([item[state_feature].unsqueeze(0), item[action_feature]], dim = 0)
                
                item.pop(action_feature)

        out_item = {}

        for target_key, convert_info in self.key_reverse_mapping.items():
            if isinstance(convert_info, dict) and convert_info['target_key'] in item:
                out_item[target_key] = item[convert_info['target_key']]
                continue

            if isinstance(convert_info, list):
                convert_info = sorted(convert_info, key=lambda x: x['end'])
                concat_list = []
                convert_success = True
                for _convert_info in convert_info:
                    if _convert_info['target_key'] not in item:
                        raise ValueError(f"{_convert_info['target_key']} is not contained in robot config as target feature")
                    concat_list.append(item[_convert_info['target_key']][..., _convert_info['target_start']:_convert_info['target_end']])
                if convert_success: out_item[target_key] = torch.cat(concat_list, dim=-1)

        for feature in self.feature_to_keep:
            if feature in item:
                out_item[feature] = item[feature]
        del item
        return out_item


    def _to_xyzw(self, pose):
        """Dataset convention -> XYZW, which ee_pose_transform works in."""
        if self.quat_convention == 'wxyz':
            return quat_wxyz_to_xyzw(pose)
        return pose

    def _from_xyzw(self, pose):
        """XYZW -> dataset convention. Inverse of `_to_xyzw`."""
        if self.quat_convention == 'wxyz':
            return quat_xyzw_to_wxyz(pose)
        return pose

    def apply(self, item, policy_eval=False):
        w_action = not policy_eval
        if w_action:
            item['action_is_pad'] = item[f"{self.org_features['actions'][0]}_is_pad"] if not len(self.actions_convert_from_state)>0 else item[f"{self.org_features['states'][0]}_is_pad"][1:]
        else:
            item['action_is_pad'] = torch.zeros(self.chunk_size)
        item = self.convert_features(item, w_action=w_action)


        for action_feature in self.actions:
            if self.action_subtract_state[action_feature] and w_action:
                state_feature = action_feature.replace('action.', 'observation.state.')
                if not (action_feature in item and state_feature in item):
                    raise ValueError(f"{action_feature} or/and {state_feature} are not in the item")
                relative_type = self.action_relative_type.get(action_feature)
                if _is_quaternion_relative_type(relative_type):
                    assert 'end.position' in action_feature
                    # relative_pose_quaternion is XYZW; convert in and back out so
                    # the delta stays in the dataset's own convention.
                    item[action_feature] = self._from_xyzw(relative_pose_quaternion(
                        self._to_xyzw(item[action_feature]),
                        self._to_xyzw(item[state_feature]),
                        relative_type=relative_type,
                    ))
                else:
                    item[action_feature] -= item[state_feature]

        if self.normalizer is not None:
            item = self.normalizer.normalize(item)

        if self.return_item_befor_padding:
            return item

        batch_dict = self.pad_and_concat(item, w_action)
        if self.wrist_augment and not policy_eval:
            augment_wrist_views(
                [batch_dict["image"], batch_dict.get("future_image", {})], self.wrist_view_keys
            )

        state = prepare_state(batch_dict, self.model_config.max_state_dim)
        actions = prepare_action(batch_dict, self.model_config.max_action_dim)
        return_image_grid_thw = getattr(self.model_config, "return_image_grid_thw", False)
        if not self.disabled_image_features:
            (
                images,
                img_masks,
                pil_images,
                image_grid_thw,
                image_augment_params,
            ) = prepare_images(
                self.processor.image_processor,
                batch_dict,
                image_keys=self.feature_config.images,
                train=self.image_augment and not policy_eval,
                use_depth_align=self.use_depth_align,
                return_image_grid_thw=return_image_grid_thw,
                return_augment_params=True,
            )
            if self.use_future_image and len(batch_dict.get("future_image", {})) > 0:
                future_obs = {**batch_dict, "image": batch_dict["future_image"]}
                future_images, _, future_pil_images, _ = prepare_images(
                    self.processor.image_processor, future_obs,
                    image_keys=self.feature_config.images,
                    train=self.image_augment and not policy_eval,
                    use_depth_align=self.use_depth_align,
                    return_image_grid_thw=False,
                    augment_params=image_augment_params,
                )
            else:
                future_images, future_pil_images = None, None
            raw_images = batch_dict.get("raw_image")
            if raw_images:
                # Teachers see the unmasked cam_high (slot 0) with the same colour
                # augmentation as the masked view; the policy inputs are unchanged.
                pil_images, future_pil_images = self._swap_in_raw_teacher_images(
                    raw_images, pil_images, future_pil_images,
                    image_augment_params if (self.image_augment and not policy_eval) else None,
                )
        else:
            images, img_masks, pil_images, image_grid_thw = [], [], [], None
            future_images, future_pil_images = None, None
            raw_images = None

        merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        image_token_count = compute_image_token_count(
            images,
            image_grid_thw=image_grid_thw,
            merge_size=merge_size,
            use_vision_boundaries=getattr(self.model_config, "qwen3vl_use_vision_boundaries", True),
        )
        batch_dict["image_token_count"] = image_token_count
        
        lang_tokens, lang_masks = prepare_language(self.model_config, self.tokenizer, batch_dict) # bs, seq_len
        action_is_pad = batch_dict['action_is_pad']

        state_joint_mask = batch_dict['state_joint_mask']
        assert self.model_config.max_state_dim >= state_joint_mask.shape[-1], f"max_action_dim is smaller than the state joint dimension: {self.model_config.max_action_dim} < {state_joint_mask.shape[-1]}"
        state_joint_mask = F.pad(state_joint_mask, (0, self.model_config.max_state_dim - state_joint_mask.shape[-1])).to(dtype=torch.bool)

        action_joint_mask = batch_dict['action_joint_mask']
        assert self.model_config.max_action_dim >= action_joint_mask.shape[-1], f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {action_joint_mask.shape[-1]}"
        action_joint_mask = F.pad(action_joint_mask, (0, self.model_config.max_action_dim - action_joint_mask.shape[-1])).to(dtype=torch.bool)
        
        chunk_joint_mask = batch_dict['chunk_joint_mask']
        assert self.model_config.max_action_dim >= chunk_joint_mask.shape[-1], f"max_action_dim is smaller than the action joint dimension: {self.model_config.max_action_dim} < {chunk_joint_mask.shape[-1]}"
        chunk_joint_mask = F.pad(chunk_joint_mask, (0, self.model_config.max_action_dim - chunk_joint_mask.shape[-1])).to(dtype=torch.bool)
        
        del batch_dict
        batch_dict = {
                'images': images,
                'img_masks': img_masks,
                'state': state,
                'lang_tokens': lang_tokens,
                'lang_masks': lang_masks,
                'actions': actions,
                'action_is_pad': action_is_pad,
                'joint_mask': chunk_joint_mask,
                'state_joint_mask': state_joint_mask,
                'action_joint_mask': action_joint_mask,
            }
        if image_grid_thw is not None:
            batch_dict['image_grid_thw'] = image_grid_thw

        if self.use_depth_align: 
            batch_dict['pil_images'] = pil_images
            if self.use_future_image:
                #assert future_pil_images is not None and future_pil_images is not []:
                batch_dict['future_pil_images'] = future_pil_images
        if raw_images:
            # (n_frames, 256), n_frames = 1 (current) or 2 (current, future).
            batch_dict['aux_patch_weight'] = raw_images['patch_weight']
        return batch_dict

    def _swap_in_raw_teacher_images(self, raw_images, pil_images, future_pil_images, augment_params):
        """Replace teacher image slot 0 with the raw cam_high frames."""
        def _prep(raw, ref):
            if augment_params is not None:
                raw = apply_visual_augmentation(raw, augment_params)
            raw = raw.to(dtype=ref.dtype)
            if raw.shape != ref.shape[1:]:
                raise ValueError(f"raw cam_high {tuple(raw.shape)} != teacher image {tuple(ref.shape[1:])}")
            return raw

        pil_images = pil_images.clone()
        pil_images[0] = _prep(raw_images['current'], pil_images)
        if future_pil_images is not None and 'future' in raw_images:
            future_pil_images = future_pil_images.clone()
            future_pil_images[0] = _prep(raw_images['future'], future_pil_images)
        return pil_images, future_pil_images

    def unapply(self, item):
        if not self.return_item_befor_padding:
            item = self.reverse_pad_and_concat(item)

        if self.normalizer is not None:
            item = self.normalizer.unnormalize(item)

        for action_feature in self.actions:
            if self.action_subtract_state[action_feature]:
                state_feature = action_feature.replace('action.', 'observation.state.')
                relative_type = self.action_relative_type.get(action_feature)
                if _is_quaternion_relative_type(relative_type):
                    assert 'end.position' in action_feature
                    # Mirror of apply(): the model's delta and the state are both
                    # in the dataset convention, so convert both into XYZW, add,
                    # and hand the absolute pose back in the same convention.
                    item[action_feature] = self._from_xyzw(absolute_pose_quaternion(
                        self._to_xyzw(item[action_feature]),
                        self._to_xyzw(item[state_feature]),
                        relative_type=relative_type,
                    ))
                else:
                    item[action_feature] += item[state_feature]

        item = self.reverse_features(item)
        return item

    def reverse_pad_and_concat(self, item):
        reverse_item = {}

        # In policy_eval, model output `actions` is always padded to max_action_dim
        # Pad mask with False at the tail so `actions[:, mask]` selects only the
        # real joint dims — the padded region is False and contributes nothing.
        state_joint_mask = item['state_joint_mask']
        assert state_joint_mask.shape[-1] == item['state'].shape[-1]

        action_joint_mask = item['action_joint_mask']
        assert action_joint_mask.shape[-1] == item['actions'].shape[-1]

        state = item['state'][state_joint_mask]
        action = item['actions'][:, action_joint_mask]
        
        for k in self.feature_config.joints:

            state_key = f'observation.state.{k}'
            if state_key in self.states:
                joint_dim = self.normalizer.norm_stats[state_key]['mean'].shape[-1]
                reverse_item[state_key] = state[:joint_dim]
                state = state[joint_dim:]
            del state_key
            
            action_key = f'action.{k}'
            if action_key in self.actions:
                joint_dim = self.normalizer.norm_stats[action_key]['mean'].shape[-1]
                reverse_item[action_key] = action[:, :joint_dim]
                action = action[:, joint_dim:]
            del action_key
        return reverse_item

    def pad_and_concat(self, item, w_action = True):
        images = {}
        future_images = {}

        def _to_uint8(img):
            # Keep images in uint8 [0, 255] before the Qwen-VL image processor, whose
            # `do_rescale=True` divides by 255. lerobot decodes mp4 (video dtype) cameras
            # already in uint8, but decodes parquet-embedded PNG (image dtype) cameras into
            # float32 [0, 1]; feeding the latter as-is would be rescaled a second time and
            # collapse the input to ~-1. Convert float [0, 1]-ish back to uint8, leave uint8 alone.
            if not torch.is_tensor(img) or not img.is_floating_point():
                return img
            try:
                max_val = float(img.max())
            except RuntimeError:
                max_val = 0.0
            if max_val <= 2.0:
                img = (img * 255).round().clamp(0, 255).to(torch.uint8)
            return img

        for image_key in self.feature_config.images:
            if image_key in self.images and image_key in item:
                if not self.use_future_image:
                    images[image_key] = _to_uint8(item[image_key])
                else:
                    images[image_key] = _to_uint8(item[image_key][0])
                    future_images[image_key] = _to_uint8(item[image_key][-1])

        raw_images = {}
        if self.aux_raw_image and RAW_CAM_HIGH_KEY in item:
            raw = _to_uint8(item[RAW_CAM_HIGH_KEY])
            masked_key = self.images[0]
            if not self.use_future_image:
                raw_images['current'] = raw
                masked = [images[masked_key]]
            else:
                raw_images['current'], raw_images['future'] = raw[0], raw[-1]
                masked = [images[masked_key], future_images[masked_key]]
            raw_list = [raw_images['current']] + ([raw_images['future']] if self.use_future_image else [])
            keep = derive_keep_mask(torch.stack(raw_list), torch.stack(masked), self.aux_outside_scale)
            # (n_frames, 256): foreground fraction of each 16x16 teacher patch.
            raw_images['patch_weight'] = F.adaptive_avg_pool2d(keep, 16).flatten(1)

        actions, action_joints_pad = [], []
        states, state_joints_pad = [], []

        for k in self.feature_config.joints:
            state_key = f'observation.state.{k}'

            if state_key in self.states:
                pad_len = self.feature_config.joints_max_dim[k] - item[state_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                states.append(F.pad(item[state_key], (0, pad_len)))
                state_joints_pad.append(F.pad(torch.ones(item[state_key].shape), (0, pad_len)))
            else:
                states.append(torch.zeros(self.feature_config.joints_max_dim[k]))
                state_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del state_key

            action_key = f'action.{k}'
            if action_key in self.actions and w_action:
                pad_len = self.feature_config.joints_max_dim[k] - item[action_key].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                actions.append(F.pad(item[action_key], (0, pad_len)))
                action_joints_pad.append(F.pad(torch.ones(item[action_key].shape[-1]), (0, pad_len)))
            elif action_key in self.actions and not w_action:
                assert action_key in self.normalizer.norm_stats, f"{action_key} not in norm keys: {self.normalizer.norm_stats.keys()}"
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                pad_len = self.feature_config.joints_max_dim[k] - self.normalizer.norm_stats[action_key]['mean'].shape[-1]
                assert pad_len >= 0, f"pad_len is negative: {pad_len}"
                action_joints_pad.append(F.pad(torch.ones(self.normalizer.norm_stats[action_key]['mean'].shape[-1]), (0, pad_len)))
            else:
                actions.append(torch.zeros(self.chunk_size, self.feature_config.joints_max_dim[k]))
                action_joints_pad.append(torch.zeros(self.feature_config.joints_max_dim[k]))
            del action_key

        action_joint_mask = torch.cat(action_joints_pad, dim=-1).to(dtype=torch.bool)
        state_joint_mask = torch.cat(state_joints_pad, dim=-1).to(dtype=torch.bool)
        state = torch.cat(states, dim=-1).to(torch.float32)
        action = torch.cat(actions, dim=-1).to(torch.float32)
        chunk_joint_mask = action_joint_mask.clone().unsqueeze(0).repeat(self.chunk_size, 1)

        batch_dict =  {
            "image": images,
            "future_image": future_images,
            "state": state,
            "action": action,
            "action_is_pad": item['action_is_pad'],
            'chunk_joint_mask': chunk_joint_mask,
            "action_joint_mask": action_joint_mask,
            "state_joint_mask": state_joint_mask,
            "prompt": [item["task"]],
        }
        if "future_video_effective_fps" in item:
            batch_dict["future_video_effective_fps"] = item["future_video_effective_fps"]
        if raw_images:
            batch_dict["raw_image"] = raw_images

        return batch_dict
