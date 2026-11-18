# Action space: joint vs end-effector, absolute vs relative

Two orthogonal config fields select the robot state/action representation used
for both training and inference:

- `data.action_space` — **which columns** the policy reads and writes: joint
  positions or end-effector poses.
- `data.action_type` — **absolute or relative** targets: the pose itself, or the
  pose minus the current state.

Together they compose into four representations, each with its own robot config:

| `action_space` | `action_type` | meaning | robot config read |
| --- | --- | --- | --- |
| `joint` (default) | `abs` | absolute joint positions | `configs/robot_configs/<robot>.yaml` |
| `joint` | `rel` | joint deltas (`action - state`) | `configs/robot_configs/<robot>_rel.yaml` |
| `ee` | `abs` | absolute end-effector poses | `configs/robot_configs/<robot>_ee.yaml` |
| `ee` | `rel` | end-effector deltas | `configs/robot_configs/<robot>_ee_rel.yaml` |

```yaml
data:
  action_space: joint   # joint | ee
  action_type: rel      # abs | rel
```

The pair is resolved in one place,
`lingbotvla.data.vla_data.utils.resolve_robot_config_path`, which both the
training dataset and the deploy server call. A missing `action_type` is **not**
treated as `abs` blindly: the representation is inferred from the robot config's
own `subtract_state` (see *Inference and backwards compatibility* below).

## How relative is implemented

`action_type` is a declaration and a guard; the mechanism lives in the robot
config's per-feature `subtract_state`:

```yaml
actions:
  - action.arm.position:
      origin_keys: [...]
      subtract_state: True     # this is what makes it relative
      relative_type: world     # REQUIRED for joint poses -- see below
  - action.effector.position:
      origin_keys: [...]
      subtract_state: False    # gripper stays absolute even in a relative run
```

With `subtract_state: True`, `FeatureTransform.apply` computes `action - state`
at load time and `FeatureTransform.unapply` adds the state back at inference, so
the model sees deltas while the simulator still receives absolute targets. The
gripper cannot be subtractive (`get_feature_mapping` asserts this for effector
features), so relative runs keep it absolute.

`relative_type` must be set explicitly for joint poses. The parser defaults it
to `quaternion_local`, and any of `{quaternion, quaternion_world,
quaternion_local}` routes through rotation composition instead of elementwise
subtraction and asserts `end.position` is present — which a joint feature never
is. `FeatureTransform.check_action_type` now rejects that pairing at startup.

## Norm stats

A relative run needs statistics computed **on the deltas**. Reusing an absolute
stats file is the one silent failure mode of this setup: with
`bounds_99_woclip`, deltas would be normalized against a quantile range roughly
two orders of magnitude too wide and collapse towards zero, with no error raised.

```bash
python scripts/compute_norm_stats.py \
  --config configs/vla/norm_compute/post_data.yaml \
  --data.data_name robotwin \
  --data.train_path /mnt/data/liangjiaqi/yzm/RoboTwinData/lerobot_dataset/RoboTwin_lerobot_v30 \
  --data.robot_config_root ./configs/robot_configs \
  --data.action_space joint \
  --data.action_type rel \
  --data.norm_path assets/norm_stats/robotwin_rel.json
```

`--data.action_type` is required here. The robot config is resolved *before*
`FeatureTransform` could infer the type, so omitting it loads the absolute
config and silently accumulates absolute statistics; the script prints the
resolved `action_type`, the `subtract_state` map, and whether it is accumulating
DELTA or ABSOLUTE statistics so the log shows which happened.

A relative run must point **both** norm-stats references at the new file, since
they are read by different code paths:

- the `norm_stats:` key inside the robot config (`<robot>_rel.yaml`) — read by
  training;
- `data.norm_stats_file` in the training config — saved into
  `lingbotvla_cli.yaml`, and read back by the deploy server.

If they disagree, a run trains on one set of statistics and is served with
another.

## Train/inference consistency

`action_space` and `action_type` are written into the run's
`lingbotvla_cli.yaml` by `save_args`. `LingbotVLAv2Server.load_vla` reads them
back from that file, so the eval server always uses the representation the
checkpoint was trained with. Neither is an eval command-line flag — there is no
way to serve a joint checkpoint as EE, or an absolute checkpoint as relative.
`unapply` returns absolute actions regardless, so relative checkpoints need no
special handling at the RoboTwin side.

Guards:

- `FeatureTransform.check_action_space` rejects a config/space mismatch (e.g.
  `action_space: ee` pointed at a joint robot config) at startup instead of
  silently training on the wrong representation.
- `FeatureTransform.check_action_type` rejects a config/type mismatch when
  `action_type` is declared explicitly, and rejects the quaternion/ joint-pose
  pairing in all cases.
- `resolve_robot_config_path` raises `FileNotFoundError` when the variant for a
  given pair does not exist, naming the four names it knows about.

## Inference and backwards compatibility

`action_type` was added after the RoboTwin v3.0 configs were first trained, so
`lingbotvla_cli.yaml` files written before it (commit `7afe3fe` and earlier) have
neither `action_space` nor `action_type`. If the field is *absent*, the type is
inferred from the robot config's `subtract_state` rather than defaulted to `abs`,
which means:

- the `7afe3fe` checkpoint resolves to `joint` + `abs`, reading `robotwin.yaml`
  and `assets/norm_stats/robotwin.json` — byte-identical to its behaviour when
  trained, so its weights load and infer unchanged;
- configs that express relative purely through `subtract_state` with no
  `action_type` (for example `configs/robot_configs/agilex_cobot_magic.yaml`)
  keep working rather than being forced into `abs`.

Only an explicitly declared `action_type` is validated. Note that
`agilex_cobot_magic.yaml` is separately broken for an unrelated reason: it sets
`subtract_state: true` on a joint pose without an explicit `relative_type`, so it
inherits the `quaternion_local` default and trips the quaternion/joint guard. Set
`relative_type: world` there to use it.

## Layout expected by `robotwin_ee.yaml`

A flat 16-dim vector per timestep, matching the packing RoboTwin's
`take_action(action, action_type='ee')` consumes:

```
 0: 3   left  ee xyz
 3: 7   left  ee quaternion
 7: 8   left  gripper
 8:11   right ee xyz
11:15   right ee quaternion
15:16   right gripper
```

This maps onto the unified feature space as `end.position` (14 = 2 x 7) and
`effector.position` (2), so `data.joints` and the model's
`action_dim` / `max_state_dim` are unchanged from the joint recipe.

## The RoboTwin EE dataset

`RoboTwin_lerobot_v30_ee` is the EE counterpart of `RoboTwin_lerobot_v30`. It
differs in exactly one respect: `observation.state` and `action` hold the 16-dim
EE vector instead of the 14-dim joint vector. **The column names are the same** as
the joint datasets — only the width changes — so the two are drop-in
alternatives. Frames, episode order, task strings, timestamps and the encoded
videos are identical (the videos are hardlinked, not re-encoded).

As in the joint data, `action[t] == observation.state[t+1]`, so actions need no
shifting.

It is built by `eework/convert_v30_to_ee.py` from the raw `demo_clean.zip`
episodes. Note that `tools/build_ee_dataset.py` is **deprecated**: it writes
separate `observation.state.ee` / `action.ee` columns, and its episode-to-raw
mapping (`e // 50`, `e % 50`) is wrong because the v30 task-block order is a
permutation of `sorted(task_dirs)`. The correct mapping has to be derived by
content.

## What `action_space: ee` needs

1. **The EE dataset** above, at `data.train_path`.
2. **Norm stats for `end.position`.** `assets/norm_stats/robotwin.json` only holds
   `arm.position` / `effector.position`. Generate
   `assets/norm_stats/robotwin_ee.json` (abs) and
   `assets/norm_stats/robotwin_ee_rel.json` (rel) with
   `scripts/compute_norm_stats.py`, passing **both** `--data.action_space ee` and
   the matching `--data.action_type`. The exact commands are in the headers of
   `configs/robot_configs/robotwin_ee.yaml` and `robotwin_ee_rel.yaml`.
3. **Eval-time EE state.** The deploy server's `_build_ee_state` assembles
   `observation.state` (16-dim) from RoboTwin's endpose fields
   (`left_endpose`, `left_gripper`, `right_endpose`, `right_gripper`) when they
   are present in the observation, so the eval client only has to forward
   `obs['endpose']`. `data_type.endpose` is already `true` in
   `task_config/demo_clean.yml` and `demo_randomized.yml`. A legacy
   `observation.state.ee` key is still accepted and normalised onto
   `observation.state`.
4. **EE actions applied as EE.** RoboTwin's `take_action` defaults to
   `action_type='qpos'`; an EE policy must pass `action_type='ee'`, which routes
   the target pose through IK/motion planning instead of joint TOPP.

   Both of the sim-side pieces are wired up, behind one flag. Pass
   `--action_space ee` to `experiment/robotwin/start_robotwin_infer_and_eval.sh`
   (or set `ACTION_SPACE=ee`); it is forwarded to the eval client as an override.
   With the flag set the client forwards `obs['endpose']` under the four flat
   endpose keys instead of the 14-dim `joint_action.vector`, and every
   `take_action` call becomes `take_action(act, action_type='ee')`. With it unset
   the joint path is byte-identical to before. The launcher rejects any value
   other than `joint`/`ee` at startup.

   A typo is worth guarding against because the failure is not obvious from the
   inside: an EE checkpoint fed a 14-dim joint vector fails episode after
   episode in a way that looks like a policy problem. The server does not need
   the flag — it reads `action_space` from the checkpoint's own training config
   and builds the 16-dim state itself — but keeping the two in sync is what makes
   the state the client sends match the one the server assembles.

   The flag is per-run, not per-checkpoint: evaluating a joint checkpoint still
   needs the default, even on a machine that has evaluated an EE one.

## Training configs

| run | config |
| --- | --- |
| joint + abs | `configs/vla/robotwin/robotwin_v30_formal.yaml` (also `_abs`) |
| joint + rel | `configs/vla/robotwin/robotwin_v30_formal_rel.yaml` |
| ee + abs | `configs/vla/robotwin/robotwin_v30_formal_abs_ee.yaml` |
| ee + rel | `configs/vla/robotwin/robotwin_v30_formal_rel_ee.yaml` |

All four keep `data.joints` identical (`arm.position: 14`, `end.position: 14`,
`effector.position: 2`). That list is the fixed **slot layout** of the 55-dim
padded state/action vector, not a declaration of which features the dataset
carries: an EE run fills the `end.position` and `effector.position` slots and
leaves `arm.position` zero-padded, exactly as a joint run leaves `end.position`
zero. Changing it would desync `action_dim: 55` from the pretrained weights.

## Quaternion convention

Two conventions meet here, and mixing them up is silent:

- **RoboTwin, and therefore `RoboTwin_lerobot_v30_ee`, is WXYZ** (scalar first).
  Its endpose comes from `transforms3d.quaternions.mat2quat` via
  `envs/robot/robot.py::_trans_endpose`, and `transforms3d`'s `quat2mat` unpacks
  `w, x, y, z = q`.
- **`lingbotvla/data/vla_data/ee_pose_transform.py` is XYZW** (scalar last):
  `quat_canonicalize` treats `q[..., 3]` as the scalar and `quat_inverse` negates
  `q[..., :3]`.

The bridge is the robot config's `quat_convention` key:

```yaml
quat_convention: wxyz   # what the DATASET stores; default 'xyzw'
```

`FeatureTransform.apply` / `unapply` convert to XYZW before calling
`relative_pose_quaternion` / `absolute_pose_quaternion` and convert back
afterwards (`quat_wxyz_to_xyzw` / `quat_xyzw_to_wxyz`, both in
`ee_pose_transform.py`). Everything outside those two calls — the parquet, the
deploy server's `observation.state`, and the action chunk sent to RoboTwin — stays
in the dataset's own convention.

The **absolute** path never composes rotations, so it is unaffected either way:
the quaternion is regressed componentwise and handed back in the same order.

For the **relative** path, getting this wrong does not raise. A
relative → absolute round trip is self-consistent under either reading (verified:
error ~1e-12 even with the wrong convention), so the only symptom is a policy
trained on geometrically wrong targets. Measured on a 0.35 rad rotation
difference, reading WXYZ as XYZW gives a relative rotation of
`[-0.153, 0, -0.083, 0.985]` where the truth is `[0.985, 0, 0, 0.174]`; under
`relative_type: quaternion_local` the translation delta additionally comes out
axis-permuted and sign-flipped. Because a round trip cannot detect this, any test
here must compare against rotation-matrix arithmetic (`Rs^T @ Ra` for the
rotation, `Rs^T @ (pa - ps)` for the local translation).

Two safeguards: an unrecognised `quat_convention` value raises at startup, and a
relative EE run that leaves the key unset gets a warning naming the assumed
convention.

Across the raw episodes sampled while adding this switch, the endpose
quaternions had a consistently non-negative `w` and no sign flips between
consecutive frames, so absolute quaternion regression is well-posed without
extra canonicalization. Relative targets are canonicalized by
`quat_canonicalize` regardless.
