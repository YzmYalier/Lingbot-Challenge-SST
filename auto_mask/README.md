# 离线 SAM multi

正式入口为根目录的 `process_data.sh`，参数、环境、恢复方法见 [交付说明](../README.md)。

- `pipeline_lerobot.py`：100 条 seen instruction 的离线提词、逐对象 mask、Qwen judge、渲染及组装。
- `lerobot_mask_worker.py`：每个 GPU 一个进程，保留 SAM3 对象 ID。
- `pipeline.py` / `vllm_space/sam3_prompt.py`：首帧 + instruction 提词，与推理端共用。
- `mask_style.py`：离线与推理共用的亮度及白色外轮廓实现。
- `composite_lerobot.py`：已有最终 mask 的样式重渲染工具。
- `check_lerobot_sam.py`：保留的 LeRobot 抽样加载检查器；正式发布校验使用根入口 `--stage validate`。

相关绝对路径通过根目录 `base.json` 解析。不要用旧工作目录的缓存，也不要直接执行底层的子集组装来生成正式全量数据。
