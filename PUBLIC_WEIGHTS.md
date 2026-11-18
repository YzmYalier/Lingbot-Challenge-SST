# 公开权重与输入数据下载、删除及校验

在 `to_submit` 内执行：

```bash
python3 download_public_weights.py
```

也可从任意工作目录使用脚本的完整路径执行。需要 Python 3.10+、网络连接及约 103.1 GB（96.00 GiB）的模型与输入数据空间，无须安装 `torch` 或 `huggingface_hub`。已有文件通过 MD5 和 SHA-256 后直接跳过；缺失或校验失败的文件依次下载。更新已有大文件时，需要额外容纳该文件的 `.part` 临时副本。

脚本位置决定 `base.json` 的位置，随后通过 `submit_paths.py` 将配置中的相对路径解析到本包。保持 `base_path: "."`，移动整个 `to_submit` 后无需修改脚本。`public_weights.json` 和 `submit_paths.py` 必须随包保留。目标路径由清单中的 `destination.path_key` 对应 `base.json.paths`，再拼接相对文件名；没有原机器路径。`Lingbot-vla-challenge` 和 `RoboTwin`（含其下的 XPolicyLab）整个目录被排除。

## 下载来源与目录

每个文件的完整下载 URL、固定提交、字节数、MD5 和 SHA-256 均记录在 [public_weights.json](public_weights.json)。下载固定提交中的文件，不跟随 `main` 更新。

| 分组（`--only`） | 下载来源 | 当前目标 | 包含内容 |
|---|---|---|---|
| `qwen3_vl` | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `Qwen3-VL-4B-Instruct/` | 2 个权重分片、索引、配置、tokenizer 等，共 14 文件 |
| `qwen3_8` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `vllm_space/qwen3.8-27B/` | 18 个权重分片及配套文件，共 32 文件 |
| `lingbot_vla2` | [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) | `LingbotVLA2_Origin/` | 6 个策略分片、两个 teacher 及配套文件，共 23 文件 |
| 同上 | 同仓库 `depth/model.pt` | `LingbotVLA2_Origin/depth/model.pt` | LingBot-Depth teacher；使用 `paths.depth_teacher` |
| 同上 | 同仓库 `dino_video/teacher_step_10000.pth` | `LingbotVLA2_Origin/dino_video/teacher_step_10000.pth` | DINO-Video teacher，另含 `config.yaml`；使用各自配置路径 |
| `moge2` | [Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) | `moge-2-vitb-normal/` | `model.pt` 及原有配套文件，共 3 文件 |
| `sam3` | [ModelScope: facebook/sam3](https://modelscope.cn/models/facebook/sam3) | `sam3/ckpt/` | `sam3.pt` 与 `LICENSE`，共 2 文件，无需认证 |
| `robotwin_data` | [Hugging Face: TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186/lerobot_dataset) | `RoboTwinData/lerobot_dataset/` | `RoboTwin_lerobot_v21.zip`、`RoboTwin_lerobot_v30.zip`，共约 6.50 GB |

RoboTwin 及 XPolicyLab 的权重和语言嵌入不在下载或校验清单内，现有文件保留。`RoboTwinData` 下的两个原始 ZIP 作为公开输入纳入同一清单，路径分别使用 `base.json.paths.zip_v21` 和 `zip_v30`；解压或预处理生成的数据目录不在清单内。仿真网格、环境依赖、编译产物和依赖包中的测试数据，以及本包并未包含的其他策略权重均不在此次清单内。

两个 ZIP 固定到数据集提交 `3dc3b798668feb99ac61cc9086d84cbcc3d79186`。本地完整文件已与该版本公布的 SHA-256 和大小核对，实测 MD5 为：

| 文件 | 字节数 | MD5 |
|---|---:|---|
| `RoboTwin_lerobot_v21.zip` | 2,395,762,459 | `da71dc0e618b6b2554be8fd5b6423c32` |
| `RoboTwin_lerobot_v30.zip` | 4,100,134,542 | `58ccac4457d1f54041aa6a13d7ce6db4` |

## SAM3 下载

SAM3 使用 [ModelScope 的 facebook/sam3](https://modelscope.cn/models/facebook/sam3)，固定提交为 `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`。`sam3.pt` 和 `LICENSE` 均匿名下载，无需登录、访问申请或 `HF_TOKEN`，也不需要安装 ModelScope SDK：

```bash
python3 download_public_weights.py --only sam3
```

两份文件的大小及 SHA-256 与原有本地文件一致，沿用原 MD5 清单。权重仍为 SAM3 的原始 `sam3.pt`。脚本不会向 ModelScope 或其 CDN 发送 Hugging Face 令牌。

## 常用命令

```bash
# 仅离线校验；缺失或损坏时退出码为 1
python3 download_public_weights.py --check

# 查看所有 URL、目标相对路径、大小和 MD5，不读取大文件
python3 download_public_weights.py --list

# 只下载一个或多个分组
python3 download_public_weights.py --only lingbot_vla2 --only moge2

# 并发下载 Qwen；只使用已核对为相同字节的官方 ModelScope 固定版本镜像
python3 download_public_weights.py --only qwen3_8 --workers 8 --mirror modelscope

# 单独下载或校验两个输入 ZIP
python3 download_public_weights.py --only robotwin_data
python3 download_public_weights.py --check --only robotwin_data

# 只恢复一个文件；路径相对于 to_submit
python3 download_public_weights.py --file sam3/ckpt/sam3.pt

# 调整网络超时与单文件尝试次数
python3 download_public_weights.py --timeout 120 --retries 6

# 使用系统工具校验默认目录布局（需在 to_submit 内运行）
md5sum -c public_weights.md5
```

如果修改了 `base.json` 中的模型目录，请使用脚本的 `--check`；`public_weights.md5` 保存的是本次审计时的默认相对路径。

## 删除清单文件

`delete_public_weights.py` 与下载脚本共用清单。默认删除全部 76 个清单文件及对应 `.part`，包括两个输入 ZIP；可按分组选择。它不会删除目录或清单之外的文件，也不会删除已生成的 raw/SAM 数据集。

```bash
# 预览全部删除范围
python3 delete_public_weights.py --dry-run

# 只预览或删除两个 ZIP（及对应 .part）
python3 delete_public_weights.py --dry-run --only robotwin_data
python3 delete_public_weights.py --only robotwin_data

# 删除全部公开权重、配套文件和两个输入 ZIP
python3 delete_public_weights.py
```

默认逐文件下载，可用 `--workers 1..16` 并发，失败后统一汇总报告。每个目标旁的 `.part` 支持续传；服务器忽略 Range 时会从头重写临时文件。下载完成后重新读取整个文件，只有大小、MD5、SHA-256 全部符合清单才原子替换正式目标。网络中断和校验失败不会覆盖已有目标。重复执行同一条命令即可继续。运行锁防止同一个包里的两个脚本进程同时写入。

`--mirror modelscope` 使用清单 `mirrors` 字段中经过大小与 SHA-256 核对的官方镜像，
仍按原始 MD5/SHA-256 验证；没有相同字节镜像的文件继续使用原来源。
Qwen 的慢连接也可使用 `python3 tools/download_qwen_fast.py --install-tool`，
在本目录独立 Conda 环境安装 aria2 并分段下载。中断后用同一入口恢复，
以保留 `.part.aria2` 的分段记录。所有目标、环境与缓存路径均从 `base.json` 解析。

2026-11-18 本机已完整下载并验证 Qwen 的 32 个文件、SAM3 的 2 个文件及
两个数据 ZIP。此前全清单审计记录见下节；本次没有下载训练所需的全部其他模型。

运行报告写入 `base.json.paths.public_weights_state`（默认 `work/public_weights/last_run.json`），包含各文件结果。中断退出码为 130，其余失败为 1，全部成功为 0。

## 本次哈希核验（2026-10-26）

当前清单共 **76 文件、103,078,467,572 字节**：30 个模型权重文件、44 个配套文件和 2 个输入 ZIP。所有文件实际读取并计算 MD5；未加载或反序列化权重。

初次审计通过 Hugging Face API 核对了全部 74 个模型及配套文件：LFS 文件公布 SHA-256，普通文件公布 Git blob SHA-1，没有官方 MD5 字段。因此先将本地文件的完整 SHA-256／Git blob SHA-1 和大小与固定版本的官方元数据逐一核对，再把相同字节计算出的 MD5 固定到清单。**清单中的 MD5 是实测值，不是将 HF ETag 当成 MD5，也不是声称上游公布了 MD5。** SAM3 的 2 文件改用 ModelScope 后，SHA-256 和大小再次核对一致；随后新增的 2 个 ZIP 也已与 Hugging Face 数据集元数据核对通过。当前共 74 文件从 Hugging Face 下载、2 文件从 ModelScope 下载。

独立迁移测试把脚本、配置、解析器和清单复制到一个空目录，从不同工作目录执行，实际下载 419,110,160 字节的 MoGe `model.pt`，MD5 为 `1f3d11493af7b2099216f21e3b865700`，与清单一致。没有为核验而重复下载全部约 96.6 GB。

SAM3 切换 ModelScope 后，在清除 Hugging Face／ModelScope 令牌的环境中，从空目录成功下载并校验 `LICENSE`，验证了 `sam3.pt` 的匿名下载和中断后的续传。下载端持续限速后主动停止额外传输，已下载前缀与本地对应字节的 MD5/SHA-256 一致；具体字节数记录在 `modelscope_download_report.json`。未声称此次重新下载完整 SAM3 权重；其完整 SHA-256 和大小已通过 ModelScope 元数据与此前完整校验的本地文件核对一致。

证据保存在 [validation/public_weights/](validation/public_weights/)：

- `summary.json`、`local_audit.json`：来源、固定版本、实测哈希与逐文件匹配结果。
- 各 Hugging Face 仓库 JSON：初次读取的官方元数据；`modelscope_sam3_*.json`：切换来源后读取的 ModelScope 元数据及匿名请求记录。
- `check.log`、`check_report.json`：移除 RoboTwin 前的 110 文件校验记录，当前清单中的 74 个模型及配套文件全部包含其中且哈希一致。`scope_update.json` 记录当时的范围调整；旧的 GitHub 嵌入审计文件仅保留为历史证据，不参与下载。新增两个 ZIP 的元数据、哈希和验证记录位于 [validation/public_dataset_zips/](validation/public_dataset_zips/)。
- `download_smoke.log`、`download_smoke_report.json`：迁移后真实下载测试。
- `modelscope_download.log`、`modelscope_download_report.json`：SAM3 匿名下载、续传、前缀哈希比对及测试范围。
- `tests.log`：下载器原有 CPU 测试记录。加入 ZIP 后的测试记录位于 `validation/public_dataset_zips/tests.log`，另检查了数据分组的删除计划只包含两个 ZIP 及其 `.part`。

重新运行逻辑测试：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_public_weights.py -v
```
