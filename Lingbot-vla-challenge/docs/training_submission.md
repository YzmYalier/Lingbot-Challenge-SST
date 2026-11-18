# 提交包训练

从 `to_submit` 执行。两条分支共用 `base.json`、`submit_paths.py`、`tools/` 和包内模型；必须上传整个提交包，单独克隆本仓库不包含这些资源。原工作目录不参与运行，Conda 可以放在包外。

```bash
# 数据处理环境与 VLA 环境当前共用 environments/vla/bin/python。
# 如果使用外部 Conda，只需在包根配置对应解释器：
python3 tools/configure.py --vla-python /path/to/conda/env/bin/python --data-python /path/to/conda/env/bin/python

# 完整数据处理：2500 episodes，548893 frames，100 个顶部 instruction 槽位。
bash process_data.sh

# 分支一：module LR；使用生成的 framefixed 数据，保留该分支原训练算法。
git -C Lingbot-vla-challenge switch feat/module-lr
bash Lingbot-vla-challenge/train_multinode.sh --check
bash Lingbot-vla-challenge/train_multinode.sh

# 分支二：module LR + action padding mask / 数据完整性修复，默认关闭 wrist augmentation。
git -C Lingbot-vla-challenge switch fix/module-lr-wrist-aug-data-integrity
bash Lingbot-vla-challenge/train_multinode.sh --check
bash Lingbot-vla-challenge/train_multinode.sh
```

`training_profile.json` 随分支指定默认 YAML；运行时不依赖 Git：

| 分支 | `configs/vla/robotwin/` 下的 YAML |
|---|---|
| `feat/module-lr` | `robotwin_v30_formal_abs_sam_judge_0.1_multi_rawteacher_modlr.yaml` |
| `fix/module-lr-wrist-aug-data-integrity` | `robotwin_v30_formal_abs_sam_judge_0.1_multi_rawteacher_modlr_wristaug_datafix.yaml` |

datafix 默认配方设置 `data.wrist_augment: false`；YAML 文件名保留以兼容已有命令。若需开启，传入 `--data.wrist_augment true`，同时使用独立的 `--train.output_dir` 区分实验。

YAML 中的 `submission://名称` 引用 `base.json.paths`。也支持 `YAML_FILE=...` 或 `--config ...` 指定包内 YAML。`train.sh tasks/vla/train_lingbotvla.py <yaml>` 和 `train_script.sh` 转发到同一入口。请使用这些入口解析 YAML，不要直接把含 `submission://` 的文件交给 torchrun。

数据路径与原工作区中的 `RoboTwinData` 相对布局一致：

```text
RoboTwinData/lerobot_dataset/
├── RoboTwin_lerobot_v21.zip
├── RoboTwin_lerobot_v30.zip
├── RoboTwin_lerobot_v30_framefixed/                         # paths.raw
└── RoboTwin_lerobot_v30_framefixed_sam_judged_multi_outline0.1_w2/  # paths.sam_output
    ├── data/chunk-*/file-*.parquet
    ├── meta/episodes/chunk-*/file-*.parquet
    ├── meta/sam_info.json
    ├── sam_masks/
    └── videos/<video_key>/chunk-*/file-*.mp4
```

两条分支都读取 `paths.sam_output`，teacher 使用其 `meta/sam_info.json.source` 对应的 `paths.raw`。搬动整个包后，已生成的这个 SAM/raw 配对会重新定位到包内；无须寻找原绝对目录。入口会检查完整 episode ID、帧数、100 个有序槽位、所有视频/数值文件及 SAM/raw 轨迹边界，拒绝缺失数据或 smoke 子集。这是启动检查；数据生成后的完整数值、指令顺序、mask 和解码验证仍由 `process_data.sh` 的 validate 阶段负责。

初始策略位于 `LingbotVLA2_Origin/`，包含 6 个 safetensors 分片、depth 和 DINO-video teacher。Qwen3-VL backbone、MoGe 和归一化文件也通过 `base.json` 选择。包内提供的是 post-training 所需初始权重，不是随机初始化训练。

输出默认写入 `Lingbot-vla-challenge/output/` 下两个独立目录：

- module LR：`robotwin_v30_formal_abs_sam_judge_0.1_multi_rawteacher_modlr_framefixed`
- datafix（默认关闭 wrist augmentation）：`robotwin_v30_modlr_no_wristaug_datafix`

已有评测 checkpoint 的目录不会被训练覆盖。保留原配方的 `enable_resume: true`，同一训练输出目录会按原 checkpointer 逻辑续训。用 `--train.output_dir submission://training_output/<新名字>` 开新实验。

默认 micro batch 为 32、global batch 为 1024。请根据实际 GPU 情况配置并行拓扑、shard 和 batch。单机/多机保持 global batch 不变，入口根据最终 CLI 覆盖值计算 accumulation 和 replicate size；不整除时直接报错。多机在每个节点运行同一命令，设置 `NNODES`（也接受 `HOST_NUM`）、`NODE_RANK`、`MASTER_ADDR` 和同一个 `MASTER_PORT`，并确保数据/权重可访问、输出为共享目录。`NPROC_PER_NODE` 优先，其后依次为 `CUDA_VISIBLE_DEVICES`、`HOST_GPU_NUM`、自动 GPU 数量。启动前先预览最终配置：

```bash
bash Lingbot-vla-challenge/train_multinode.sh --dry-run
```

`EXTRA_ARGS='--train.max_steps 10 --train.use_wandb false'` 仍可用，也可直接附加 CLI 参数。训练日志与展开后的 YAML 写入 `work/training/logs/`，按节点与每次启动区分；失败返回非零状态码。缓存、临时文件、TensorBoard/W&B 产物均留在提交包内。W&B 默认 offline；需要上传时显式设置 `WANDB_MODE=online` 和 `WANDB_API_KEY`，入口不读取脚本中的旧密钥。

`--dry-run` 只预览最终配置；`--check-models` 可在数据处理完成前检查模型资源；`--check` 检查全量数据和模型。无 GPU 的检查机器需按计划使用的拓扑设置 `NPROC_PER_NODE`。当前环境的 torchcodec 缺少 FFmpeg 动态库，项目已有的 PyAV 回退可读取视频，入口不会修改外部 Conda 环境。

本次验证记录在包根 `validation/training/`。CPU 测试和真实 smoke 视频的 loader/transform/collator 检查不能替代全量 GPU 训练；完整数据尚需通过 `process_data.sh` 生成。
