# VLA SFT in Practice

[![Blog](https://img.shields.io/badge/Blog-3B82F6?style=flat-square&logo=rss&logoColor=white)](https://yzmyalier.github.io/Lingbot-Challenge-SST/)
![Datasets — Coming soon](https://img.shields.io/badge/Datasets-Coming%20soon-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E&labelColor=334155)
![Checkpoints — Coming soon](https://img.shields.io/badge/Checkpoints-Coming%20soon-F59E0B?style=flat-square&logo=huggingface&logoColor=FFD21E&labelColor=334155)

**A Competition Report for the Robbyant Embodied AI Challenge**

English | [Chinese](README.zh-CN.md)

This repository contains the data processing, training, and evaluation code for our RoboTwin experiments with LingBot-VLA 2.0. The final recipe combines Qwen–SAM3 foreground inputs, mask review, raw-image teacher supervision, foreground-weighted distillation, module-wise learning rates, and data integrity fixes.

[Installation](#installation) · [Quick start](#quick-start-with-preprocessed-data) · [Datasets](#datasets) · [Full data rebuild](#rebuild-from-lerobot-v21-and-v30) · [Evaluation](#evaluation)

## Project resources

| Resource | Status |
|---|---|
| Project page and technical report | [Blog](https://yzmyalier.github.io/Lingbot-Challenge-SST/) |
| Repaired LeRobot data and precomputed SAM masks | Download links to be added after full validation and upload |
| Trained checkpoints | Release link to be added separately |

Full Qwen–SAM3 preprocessing is time-consuming. We are releasing the repaired LeRobot data together with precomputed SAM masks and rendered training videos so that users can start training without repeating mask generation. Training with these preprocessed data does not require running the Qwen or SAM3 services.

## Method

1. **Extract object phrases.** Qwen reads the instruction and the first camera frame to identify relevant objects.
2. **Segment and track.** SAM3 follows each object phrase through the episode.
3. **Review object masks.** Qwen judges candidate objects using the original frame and a highlighted mask image.
4. **Build policy inputs.** The top camera view preserves the accepted foreground, dims the background, and adds a white outline. The wrist views remain unchanged.
5. **Fine-tune the policy.** The policy receives masked images, while the depth and video teachers receive aligned raw images. The final recipe also applies foreground-weighted distillation, module-wise learning rates, and action-padding masks.

The final recipe disables wrist color augmentation, even though its configuration filename contains `wristaug`.

### Reported results

| Configuration | Training step | Clean success | Randomized success |
|---|---:|---:|---:|
| Absolute-joint baseline | 15,000 | 77.10% | 36.20% |
| Final recipe: N5 | 30,000 | 81.86% | 58.22% |

These results are from the accompanying competition report. Both configurations use clean training demonstrations and are evaluated on 50 tasks with 100 episodes per task in each setting.

## Installation

Create the required Conda environments by following the official projects:

| Environment | Official instructions | Role in this repository |
|---|---|---|
| LingBot-VLA 2.0 | [Official repository and installation guide](https://github.com/Robbyant/lingbot-vla-v2#installation) | Policy training, data loading, and policy inference |
| RoboTwin | [Official repository](https://github.com/RoboTwin-Platform/RoboTwin) and [installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) | Simulation and evaluation; follow the guide to prepare its assets |
| SAM3 | [Official installation guide](https://github.com/facebookresearch/sam3#installation) | Offline mask generation and online mask inference |
| vLLM | [Official repository](https://github.com/vllm-project/vllm) and [full source-build instructions](https://docs.vllm.ai/en/v0.20.2/getting_started/installation/gpu/#full-build-with-compilation) | Serving Qwen for phrase extraction and mask review |

For vLLM, create a dedicated Conda environment and **compile and install from repository source**. The integration is pinned to `v0.20.2`, commit `bc150f50299199599673614f80d12a196f377655`. Use that revision when following the official build instructions.

Use the upstream instructions to install dependencies, then run the entry points in this repository to use the maintained training, data, and inference changes.

Some training, inference, and data-processing interfaces may need adjustment for the GPUs available in your environment. Review device selection, parallelism, inference concurrency, and memory-related settings before running the entry points.

### Configure paths

All project paths and interpreters are resolved through [base.json](base.json) and [submit_paths.py](submit_paths.py). Update the interpreter entries for your own Conda environments before running any commands. The data-processing interpreter may share the LingBot-VLA environment when its data dependencies are available.

For example, replace the placeholder paths below:

```bash
python3 tools/configure.py \
  --vla-python /path/to/lingbotvla/bin/python \
  --data-python /path/to/lingbotvla/bin/python \
  --robotwin-python /path/to/robotwin/bin/python \
  --sam3-python /path/to/sam3/bin/python \
  --vllm-python /path/to/vllm/bin/python
```

Set `paths.vllm_source` to the source checkout used for the vLLM build. Set `paths.ffmpeg` to your FFmpeg executable, and configure the simulation runtime paths according to the RoboTwin installation. Keep data, model, and output locations in `base.json`.

The commands below assume that you are in the repository root.

## Quick start with preprocessed data

### 1. Prepare the paired datasets

After the dataset release links are available, download both the repaired raw dataset and the SAM dataset into the locations described in [Datasets](#datasets). Both are required by the default raw-teacher training recipe.

The repository includes the episode mapping and original instruction files. Check them with:

```bash
bash prepare_data_inputs.sh --check
```

After placing both datasets, update the raw-image reference in the SAM metadata to the configured local location:

```bash
python3 tools/configure.py --relocate
```

### 2. Download the pretrained training models

```bash
python3 download_public_weights.py \
  --only qwen3_vl --only lingbot_vla2 --only moge2
```

This downloads the public backbone, initial policy, and teacher assets from the pinned [model manifest](public_weights.json). It does not download our fine-tuned checkpoints or the preprocessed datasets. Existing files are checked against their recorded hashes before being reused.

### 3. Check the data and train

```bash
python3 tools/check_training_data.py
bash train.sh --check
bash train.sh
```

The default recipe is [the final datafix configuration](Lingbot-vla-challenge/configs/vla/robotwin/robotwin_v30_formal_abs_sam_judge_0.1_multi_rawteacher_modlr_wristaug_datafix.yaml). Its key settings include:

| Setting | Value |
|---|---|
| Action representation | Absolute joint |
| Action chunk length | 50 |
| SAM instruction mode | `multi` |
| Raw images for teachers | `aux_raw_image: true` |
| Foreground-weighted distillation | Enabled |
| Module-wise learning rates | Enabled |
| Action-padding mask | `mask_action_pad: true` |
| Wrist color augmentation | `wrist_augment: false` |

Use `bash train.sh --dry-run` to inspect the resolved recipe, or `bash train.sh --check-models` to check model assets before the dataset is ready.

Training writes the experiment configuration and checkpoints under `paths.training_run`. The saved `lingbotvla_cli.yaml` must stay with the experiment when its checkpoints are used for evaluation.

## Datasets

The release consists of two paired datasets:

| Dataset | Contents | Configuration key |
|---|---|---|
| Repaired LeRobot v3.0 raw | Complete camera videos, numerical state/action data, and repaired metadata | `paths.raw` |
| Judged SAM multi | Rendered top-camera views, lossless binary masks, instruction-to-mask indices, judge records, and matching training metadata | `paths.sam_output` |

Default layout:

```text
RoboTwinData/
└── lerobot_dataset/
    ├── RoboTwin_lerobot_v30_framefixed/
    └── RoboTwin_lerobot_v30_framefixed_sam_judged_multi_outline0.1_w2/
```

Each dataset preserves all **2,500 episodes** and **548,893 time steps**. The multi-instruction data retain **100 ordered seen instructions per episode**, for **250,000 instruction slots**. Slots that use the same phrase set can share a stored mask/video group; they remain separate instruction slots.

The repair uses LeRobot v2.1 videos to restore complete image sequences while preserving the v3.0 episode IDs and numerical state/action values. The original instruction order is retained. Action-padding masks are applied by the training code.

The raw dataset is also the source of the unmasked images used by the teachers. Its location is recorded in the SAM dataset's `meta/sam_info.json`; `tools/configure.py --relocate` updates this reference after moving the paired datasets.

### Mask semantics

| Parameter | Value |
|---|---|
| Phrase prompt version | `robotwin-sam3-v3.5` |
| SAM3 detection thresholds | `score_threshold_detection=0.3`, `new_det_thresh=0.5` |
| Object acceptance | Normalized `P(yes) >= 0.5` |
| Judge-image background scale | 0.2 |
| Final rendered background scale | 0.1 |
| Final outline | White, 2 pixels |
| Saved binary masks | Lossless FFV1, values 0/255; the outline is not part of the mask |

The judge-image background scale and the final rendering scale serve different purposes. Object decisions use the current and preceding frames.

## Rebuild from LeRobot v2.1 and v3.0

Use this workflow to regenerate the complete dataset from the public RoboTwin LeRobot v2.1 and v3.0 inputs. It repairs the raw data, extracts phrases for all episodes and instructions, runs SAM3 tracking and Qwen mask review, and assembles the full SAM dataset.

**Full mask generation is time-consuming. Use the precomputed dataset release for routine training.**

### 1. Download the original data and segmentation models

```bash
python3 download_public_weights.py \
  --only robotwin_data --only sam3 --only qwen3_8

bash prepare_data_inputs.sh --check
```

The [download manifest](public_weights.json) pins both original dataset ZIPs and the model files. The repository supplies the mapping and original instruction JSONs because the two LeRobot ZIPs alone do not contain the complete multi-instruction input.

### 2. Repair the raw videos

```bash
bash process_data.sh --check --stage raw
bash process_data.sh --stage raw
```

This stage matches complete state/action sequences between the source formats, restores the camera videos from v2.1, and verifies frame counts, timestamps, dimensions, and numerical data.

### 3. Start Qwen and generate all SAM masks

In one terminal, start the configured Qwen service:

```bash
python3 tools/services.py qwen
```

Keep it running while the mask stage executes. The endpoint serves `Qwen/Qwen3.8-27B` and must support two-image inputs and token log probabilities.

In another terminal, run the full mask pipeline:

```bash
bash process_data.sh --stage masks \
  --qwen-url http://127.0.0.1:8000/v1/chat/completions
```

The mask stage launches SAM3 workers automatically. It extracts object phrases from each instruction and the episode's first frame, tracks the corresponding objects through the complete repaired video, asks Qwen to review masks, and renders the accepted foreground for each instruction slot.

### 4. Assemble and verify the full dataset

```bash
bash process_data.sh --stage assemble
python3 tools/check_training_data.py
```

Assembly requires every episode to be complete and runs the output validator. The output paths are the same `paths.raw` and `paths.sam_output` used by training. Stop the manually started Qwen service when mask generation is finished and the service is no longer needed.

The combined entry point, `bash process_data.sh --stage all`, performs the same stages and manages its own Qwen service. Run it when you are not already managing a service at the configured endpoint.

Processing logs are written under `work/process_data/logs/`; mask progress is recorded in `work/process_data/sam-full-work/progress.json`. Completed work is cached for resumption. Changing the input, model, or processing code requires a new work directory. Existing published outputs are preserved.

## Evaluation

Prepare the RoboTwin environment and assets, the SAM3 and vLLM environments, the segmentation models, and the policy checkpoint before evaluation. The same public model downloader can obtain the SAM3 and Qwen assets:

```bash
python3 download_public_weights.py --only sam3 --only qwen3_8

bash eval.sh --check
bash eval.sh
```

The default entry point evaluates the checkpoint at step 30,000 under `paths.training_run`, first on `demo_randomized` and then on `demo_clean`, with 100 episodes per task. Results are kept in separate directories under `paths.eval_output`.

To select a different checkpoint:

```bash
MODEL_PATH=/path/to/experiment/checkpoints/global_step_30000/hf_ckpt bash eval.sh
```

Keep the experiment's `lingbotvla_cli.yaml` with the checkpoint. Online mask inference is maintained in [inference/](inference/); `RoboTwin/script/deploy/sam_mask/` provides compatibility imports. For standalone mask serving, use `bash start_inference_mask.sh`.

## Validation

To validate a complete paired dataset and read samples through the actual training loader:

```bash
bash process_data.sh --stage validate
python3 tools/check_training_data.py
```

The output audit checks full episode and instruction coverage, numerical data and file hashes, mask indices, and judge errors, with sampled video/mask decoding. Assembly also decodes every unique rendered segment and final mask. The training-loader check reads first/last-frame samples, including the raw teacher view.

Validation reports are written to `work/process_data/validation_report.json` and `work/sam_reuse_audit/training_loader_smoke.json`. These checks establish data integrity and loader compatibility. Policy training and benchmark evaluation are separate procedures.

## Repository layout

| Path | Purpose |
|---|---|
| `train.sh`, `eval.sh` | Final-recipe training and evaluation entry points |
| `process_data.sh`, `tools/` | Full data rebuild, configuration, service management, and validation |
| `auto_mask/` | Offline phrase extraction, tracking, judging, and rendering |
| `inference/` | Maintained online mask client and policy wrapper |
| `data_inputs/` | Episode mapping, original instructions, SAM configuration, and provenance |
| `Lingbot-vla-challenge/` | Policy implementation and experiment configurations |
| `RoboTwin/` | Simulation and evaluation integration |
| `sam3/` | Segmentation implementation and project adapters |
| `vllm_space/sam3_prompt.py` | Shared object-phrase prompt |
| `tests/` | Data integrity and downloader logic tests |

## Acknowledgments and third-party licenses

This work builds on [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), [SAM3](https://github.com/facebookresearch/sam3), Qwen, and [vLLM](https://github.com/vllm-project/vllm).

Third-party components retain their respective licenses. See the included [LingBot-VLA license](Lingbot-vla-challenge/LICENSE), [RoboTwin license](RoboTwin/LICENSE), and [SAM license](sam3/LICENSE), together with the notices in their source directories.
