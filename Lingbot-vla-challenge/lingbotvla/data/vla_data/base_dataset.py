# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import inspect
import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import hf_transform_to_torch
    LEROBOT_DATASET_API = "v3"
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import hf_transform_to_torch
    LEROBOT_DATASET_API = "v2"

from datasets import load_dataset as _hf_load_dataset

from ...utils import logging
from .utils import FeatureTransform, resolve_robot_config_path, RAW_CAM_HIGH_KEY
from .video_utils import decode_video_frames
from .submission_paths import raw_source_path
from .video_integrity import validate_video_segments, episode_video_timestamps


logger = logging.get_logger(__name__)


def _to_relative_indices(dataset, query_indices):
    index_map = getattr(dataset, "_absolute_to_relative_idx", None)
    if index_map is None:
        return query_indices
    return [index_map[idx] for idx in query_indices]


def _get_task_name(tasks, task_idx):
    if hasattr(tasks, "iloc"):
        return tasks.iloc[task_idx].name
    return tasks[task_idx]


def _resolve_lerobot_location(repo_id):
    repo_path = Path(repo_id).expanduser()
    if repo_path.exists():
        return repo_path.name, repo_path
    return repo_id, None


def _filter_supported_kwargs(callable_obj, kwargs):
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}



class LeRobotDataset(BaseLeRobotDataset):    
    def __init__(
        self,
        repo_id: str,
        load_image: bool = True,
        sam_mode: str = "auto",
        aux_raw_image: bool = False,
        video_integrity: str = "error",
        **kwargs,
    ):
        super().__init__(repo_id, **kwargs)
        self.load_image = load_image
        self.video_integrity = video_integrity
        if load_image and LEROBOT_DATASET_API == "v3":
            validate_video_segments(self.meta, video_integrity)
        self.sam_mode = sam_mode
        self._sam_video_key = "observation.images.cam_high"
        self._sam_variant_video_keys: list[str] = []
        self._sam_static_video_keys: list[str] = []
        self._configure_sam_mode()
        self.aux_raw_image = bool(aux_raw_image)
        self._raw_meta = None
        if self.aux_raw_image:
            self._configure_aux_raw_image()

    def _configure_aux_raw_image(self) -> None:
        """Pair this SAM dataset's cam_high with the unmasked source recording.

        The SAM datasets are frame-for-frame copies of ``sam_info.json["source"]``
        with only cam_high re-rendered, so the raw frame of a sample is found
        with the same episode index and episode-relative timestamp.  The
        episode lengths and cam_high segment bounds are checked once here so a
        mismatched source fails at startup instead of silently pairing wrong
        frames.
        """

        sam_info_path = Path(self.root) / "meta" / "sam_info.json"
        if not sam_info_path.exists():
            raise FileNotFoundError(
                f"aux_raw_image=True requires {sam_info_path} (it records the raw source dataset)."
            )
        with sam_info_path.open("r", encoding="utf-8") as f:
            source = raw_source_path(json.load(f).get("source"), self.root)
        if not source or not Path(source).exists():
            raise FileNotFoundError(f"{sam_info_path} source {source!r} does not exist.")
        self._raw_meta = LeRobotDatasetMetadata(Path(source).name, root=Path(source))
        if LEROBOT_DATASET_API == "v3":
            validate_video_segments(self._raw_meta, self.video_integrity)
        if self._sam_video_key not in self._raw_meta.video_keys:
            raise ValueError(f"Raw source {source} has no {self._sam_video_key} video.")

        raw_eps = self._raw_meta.episodes
        sam_eps = self.meta.episodes
        if len(raw_eps) != len(sam_eps):
            raise ValueError(
                f"Raw source has {len(raw_eps)} episodes, SAM dataset has {len(sam_eps)}."
            )
        sam_key = (
            self._sam_variant_video_keys[0] if self.sam_mode == "multi" else self._sam_video_key
        )
        raw_col = f"videos/{self._sam_video_key}/from_timestamp"
        raw_to = f"videos/{self._sam_video_key}/to_timestamp"
        sam_col = f"videos/{sam_key}/from_timestamp"
        sam_to = f"videos/{sam_key}/to_timestamp"
        for name_raw, name_sam in (("episode_index", "episode_index"), ("length", "length")):
            if list(raw_eps[name_raw]) != list(sam_eps[name_sam]):
                raise ValueError(f"Raw source and SAM dataset disagree on {name_raw}.")
        for c_raw, c_sam in ((raw_col, sam_col), (raw_to, sam_to)):
            diff = (torch.tensor(raw_eps[c_raw]) - torch.tensor(sam_eps[c_sam])).abs().max()
            if diff > 1e-4:
                raise ValueError(f"Raw source and SAM dataset cam_high {c_raw} differ by {diff}.")

    def _query_raw_cam_high(self, query_ts: list[float], ep_idx: int) -> torch.Tensor:
        """Decode the raw cam_high frames at episode-relative ``query_ts``."""

        if LEROBOT_DATASET_API == "v3":
            ep = self._raw_meta.episodes[ep_idx]
            query_ts = episode_video_timestamps(
                ep, self._sam_video_key, query_ts, self._raw_meta.fps, self.video_integrity
            )
        video_path = Path(self._raw_meta.root) / self._raw_meta.get_video_file_path(
            ep_idx, self._sam_video_key
        )
        frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
        return frames.squeeze(0)

    def _configure_sam_mode(self) -> None:
        """Configure optional RoboTwin SAM multi-video sampling.

        ``sam_multi`` keeps the normal robot configuration unchanged: the
        canonical ``observation.images.cam_high`` key is resolved to one of
        ``observation.images.cam_high_1`` ... ``_N`` for each sample.  The
        selected frame is returned under the canonical key so the existing
        feature transform and model input layout remain unchanged.
        """

        if self.sam_mode not in {"auto", "single", "multi"}:
            raise ValueError(
                f"Unsupported sam_mode={self.sam_mode!r}; expected 'auto', 'single', or 'multi'."
            )

        sam_info_path = Path(self.root) / "meta" / "sam_info.json"
        sam_info = None
        if sam_info_path.exists():
            with sam_info_path.open("r", encoding="utf-8") as f:
                sam_info = json.load(f)

        detected_mode = (sam_info or {}).get("mode", "single")
        if detected_mode not in {"single", "multi"}:
            raise ValueError(
                f"{sam_info_path} declares unsupported SAM mode {detected_mode!r}."
            )

        if self.sam_mode == "auto":
            self.sam_mode = detected_mode
        elif self.sam_mode != detected_mode:
            raise ValueError(
                f"sam_mode={self.sam_mode!r} does not match {sam_info_path} mode={detected_mode!r}."
            )

        if self.sam_mode != "multi":
            return

        if sam_info is None:
            raise FileNotFoundError(
                f"sam_mode='multi' requires {sam_info_path} so the cam_high variant mapping can be validated."
            )

        configured_video_keys = sam_info.get("video_keys")
        if not configured_video_keys:
            raise ValueError(f"{sam_info_path} has no non-empty video_keys for SAM multi data.")

        variant_by_index: dict[int, str] = {}
        pattern = re.compile(r"^observation\.images\.cam_high_(\d+)$")
        for video_key in configured_video_keys:
            match = pattern.fullmatch(video_key)
            if match is None:
                raise ValueError(
                    f"SAM multi video key {video_key!r} does not match "
                    "observation.images.cam_high_<positive integer>."
                )
            variant_index = int(match.group(1)) - 1
            if variant_index < 0 or variant_index in variant_by_index:
                raise ValueError(f"Duplicate or invalid SAM multi video key {video_key!r}.")
            if video_key not in self.meta.video_keys:
                raise ValueError(
                    f"SAM multi video key {video_key!r} is listed in sam_info.json but "
                    "is absent from meta/info.json features."
                )
            variant_by_index[variant_index] = video_key

        if not variant_by_index:
            raise ValueError("SAM multi metadata contains no usable cam_high variants.")

        expected_indices = set(range(len(variant_by_index)))
        if set(variant_by_index) != expected_indices:
            raise ValueError(
                "SAM multi cam_high variants must have contiguous numeric suffixes "
                f"starting at 1; found indices {sorted(variant_by_index)}."
            )

        self._sam_variant_video_keys = [variant_by_index[i] for i in range(len(variant_by_index))]
        self._sam_static_video_keys = [
            video_key for video_key in self.meta.video_keys
            if video_key not in self._sam_variant_video_keys
        ]

    def _get_sam_episode_tasks(self, ep_idx: int) -> list[str]:
        """Return the episode-local SAM instructions in video-key order."""

        episode = self.meta.episodes[ep_idx]
        tasks = episode.get("tasks") if hasattr(episode, "get") else episode["tasks"]
        tasks = list(tasks)
        if len(tasks) != len(self._sam_variant_video_keys):
            raise ValueError(
                f"Episode {ep_idx} has {len(tasks)} SAM instructions but "
                f"{len(self._sam_variant_video_keys)} cam_high videos."
            )
        return [str(task) for task in tasks]

    def _sample_sam_variant(self, ep_idx: int) -> tuple[int, str, str]:
        """Sample one paired cam_high video and instruction for an episode."""

        tasks = self._get_sam_episode_tasks(ep_idx)
        variant_index = random.randrange(len(self._sam_variant_video_keys))
        return (
            variant_index,
            self._sam_variant_video_keys[variant_index],
            tasks[variant_index],
        )

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        video_key_overrides: dict[str, str] | None = None,
    ) -> dict[str, list[float]]:
        """Build timestamps for the logical image keys returned to the model.

        LeRobot's implementation iterates every physical key in metadata.  A
        SAM multi episode has 100 physical high-camera keys, whereas one sample
        should expose only the selected stream under the canonical
        ``observation.images.cam_high`` key.  Querying through the logical keys
        here also lets the normal delta timestamps select current and future
        frames for the selected stream.
        """

        if self.sam_mode != "multi":
            return super()._get_query_timestamps(current_ts, query_indices)

        overrides = video_key_overrides or {}
        logical_video_keys = [self._sam_video_key, *self._sam_static_video_keys]
        query_timestamps: dict[str, list[float]] = {}
        for logical_key in logical_video_keys:
            # The delta timestamps are normally declared with the canonical
            # key.  Accept the physical key as well for callers that construct
            # a LeRobotDataset directly.
            timestamp_indices = None
            if query_indices is not None:
                timestamp_indices = query_indices.get(logical_key)
                actual_key = overrides.get(logical_key, logical_key)
                if timestamp_indices is None:
                    timestamp_indices = query_indices.get(actual_key)

            if timestamp_indices is None:
                query_timestamps[logical_key] = [current_ts]
                continue

            if self._absolute_to_relative_idx is not None:
                relative_indices = [
                    self._absolute_to_relative_idx[idx] for idx in timestamp_indices
                ]
                timestamps = self.hf_dataset[relative_indices]["timestamp"]
            else:
                timestamps = self.hf_dataset[timestamp_indices]["timestamp"]
            query_timestamps[logical_key] = torch.stack(timestamps).tolist()

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.

        Tries column-first [key][indices] for speed, falls back to row-first.

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            # Video timestamps are queried separately by ``_query_videos``.
            # In SAM multi mode the canonical robot key
            # ``observation.images.cam_high`` is an alias for one physical
            # ``cam_high_N`` stream and is not a parquet column at all.
            if key in self.meta.video_keys or key == self._sam_video_key:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = _to_relative_indices(self, q_idx)
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
        video_key_overrides: dict[str, str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            actual_video_key = (video_key_overrides or {}).get(vid_key, vid_key)
            if LEROBOT_DATASET_API == "v3":
                # LeRobot v3 stores episodes sequentially in a shared mp4, so
                # query timestamps are relative to the episode start.
                ep = self.meta.episodes[ep_idx]
                query_ts = episode_video_timestamps(
                    ep, actual_video_key, query_ts, self.fps, self.video_integrity
                )

            video_path = self.root / self.meta.get_video_file_path(ep_idx, actual_video_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            # Keep the canonical key expected by robotwin.yaml and
            # FeatureTransform, even when the physical video was cam_high_N.
            item[vid_key] = frames.squeeze(0)

        return item

    def __getitem__(self, idx) -> dict:
        # Ensure dataset is loaded when we actually need to read from it
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        sam_instruction = None
        video_key_overrides = None
        if self.sam_mode == "multi":
            _, selected_video_key, sam_instruction = self._sample_sam_variant(ep_idx)
            video_key_overrides = {self._sam_video_key: selected_video_key}
        
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
            
        if len(self.meta.video_keys) > 0 and self.load_image:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(
                current_ts,
                query_indices,
                video_key_overrides=video_key_overrides,
            )
            video_frames = self._query_videos(
                query_timestamps,
                ep_idx,
                video_key_overrides=video_key_overrides,
            )
            if self.aux_raw_image:
                # Same timestamps as the (masked) cam_high, so current/future
                # frames stay paired with the frames the policy sees.
                video_frames[RAW_CAM_HIGH_KEY] = self._query_raw_cam_high(
                    query_timestamps[self._sam_video_key], ep_idx
                )
            item = {**video_frames, **item}

        if self.image_transforms is not None and self.load_image:
            # SAM multi metadata contains all 100 cam_high_N keys, while one
            # sample intentionally contains only the selected high view plus
            # the two one-to-one wrist views.
            if self.sam_mode == "multi":
                image_keys = [self._sam_video_key, *self._sam_static_video_keys]
            else:
                image_keys = self.meta.camera_keys
            image_keys = [cam for cam in image_keys if cam in item]
            if self.aux_raw_image:
                image_keys.append(RAW_CAM_HIGH_KEY)
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        if sam_instruction is not None:
            item["task"] = sam_instruction
        else:
            task_idx = item["task_index"].item()
            item["task"] = _get_task_name(self.meta.tasks, task_idx)

        return item

class VLADataset(Dataset):
    def __init__(
        self,
        repo_id,
        data_name,
        dataset_config,
        robot_config_root,
        config=None,
        processor=None,
        video_backend = 'torchcodec',
        chunk_size = 50,
        image_size = (224, 224),
        do_nomalize = True,
        return_item = False,
        disabled_image_features = False,
        feature_transform = None,
        transform=None,
        image_augment = False,
        use_depth_align = False,
        use_future_image = False,
        action_space = 'joint',
        action_type = 'abs',
        norm_stats_file = None,
        sam_mode = 'auto',
        aux_raw_image = False,
        video_integrity = 'error',
    ):
        if do_nomalize and config is None:
            raise ValueError("VLADataset requires a model config; pass model.config via build_vla_dataset.")

        self.processor = processor
        self.config = config
        self.chunk_size = chunk_size
        self.data_name = data_name
        self.disabled_image_features = disabled_image_features
        self.use_depth_align = use_depth_align
        self.use_future_image = use_future_image

        load_image = True if do_nomalize else False

        if feature_transform is None:
            robot_config = resolve_robot_config_path(robot_config_root, data_name, action_space, action_type)
            self.feature_transform = FeatureTransform(robot_config, dataset_config, self.config, \
                        processor, disabled_image_features, do_nomalize, \
                        chunk_size=chunk_size, return_item_befor_padding=return_item,\
                        image_augment=image_augment, use_depth_align=use_depth_align,
                        use_future_image=use_future_image, action_space=action_space,
                        action_type=action_type, norm_stats_path=norm_stats_file,
                        aux_raw_image=aux_raw_image)
        else:
            self.feature_transform = feature_transform

        self.action_features = self.feature_transform.actions
        self.state_features = self.feature_transform.states
        self.image_features = self.feature_transform.images
        
        lerobot_repo_id, lerobot_root = _resolve_lerobot_location(repo_id)
        metadata_kwargs = _filter_supported_kwargs(
            LeRobotDatasetMetadata.__init__,
            {"repo_id": lerobot_repo_id, "root": lerobot_root},
        )
        self.dataset_meta = LeRobotDatasetMetadata(**metadata_kwargs)
        merged_delta = {**self.get_delta_timestamps(), **self.get_video_delta_timestamps()}

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=Resize(image_size),
            delta_timestamps=merged_delta,
            load_image=load_image,
            sam_mode=sam_mode,
            aux_raw_image=aux_raw_image and load_image,
            video_integrity=video_integrity,
        )
        if self.feature_transform.aux_raw_image and self.feature_transform.aux_outside_scale is None:
            # Same factor the masked cam_high was rendered with.
            with (Path(self.dataset.root) / "meta" / "sam_info.json").open("r", encoding="utf-8") as f:
                self.feature_transform.aux_outside_scale = float(json.load(f)["outside_scale"])

        self.return_item = return_item
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def get_features(self):
        features = set()
        for feature_category, _features in self.feature_transform.org_features.items():
            if len(self.feature_transform.actions_convert_from_state)>0 and feature_category == 'actions':
                continue
            features.update(_features)
        features.update(self.feature_transform.feature_to_keep)
        features = [x for x in list(features) if x not in ['action_is_pad', 'task', 'subtask']]
        return features

    def get_delta_timestamps(self, return_indices = False):
        delta_timestamps = {}
        fps = None if return_indices else self.dataset_meta.fps
        if not len(self.feature_transform.actions_convert_from_state)>0:
            for action_feature in self.feature_transform.org_features['actions']:
                delta_timestamps[action_feature] = [t / fps if fps else t for t in range(self.chunk_size)]
        else:
            for state_feature in self.feature_transform.org_features['states']:
                delta_timestamps[state_feature] = [t / fps if fps else t for t in range(self.chunk_size+1)]
        return delta_timestamps

    def get_video_delta_timestamps(self):
        """Multi-frame time offsets for video keys; returns an empty dict when disabled."""

        fps = self.dataset_meta.fps
        if self.use_future_image:
            offsets = [0, (self.chunk_size - 1) / fps]
            return {cam: offsets for cam in self.feature_transform.org_features['images']}
        else:
            return {}

    def check_lerobot_item(self, item):
        # if state or action is a 0-d tensor, convert it to 1-d tensor
        if not len(self.feature_transform.actions_convert_from_state)>0:
            for action_feature in self.feature_transform.org_features['actions']:
                if len(item[action_feature].shape) == 1:
                    item[action_feature] = item[action_feature].unsqueeze(-1)
            
            for state_feature in self.feature_transform.org_features['states']:
                if len(item[state_feature].shape) == 0:
                    item[state_feature] = item[state_feature].unsqueeze(-1)
        else:
            for state_feature in self.feature_transform.org_features['states']:
                if len(item[state_feature].shape) == 1:
                    item[state_feature] = item[state_feature].unsqueeze(-1)
        return item

    def getitem(self, idx):
        raw_item = self.check_lerobot_item(self.dataset[idx])
        if (
            self.use_future_image
            and "future_video_effective_fps" not in raw_item
            and hasattr(self, "dataset_meta")
        ):
            raw_item["future_video_effective_fps"] = torch.tensor(
                float(self.dataset_meta.fps) / float(max(1, self.chunk_size - 1)),
                dtype=torch.float32,
            )
        item = self.feature_transform.apply(raw_item)
        if self.transform is not None:
            item = self.transform(item, raw_item, self.feature_transform.feature_config.images, self.feature_transform.key_mapping)
        return item

    def __getitem__(self, idx):
        item = self.getitem(idx)
        return item
