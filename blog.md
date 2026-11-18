# Rethinking VLA Post-training: Change the Input, Not the Model

**改输入，不改模型：VLA 监督微调中的预训练一致性**

*LingBot 挑战赛初赛技术报告 · 基于 LingBot-VLA 2.0 的 RoboTwin 2.0 后训练实证研究*

作者：TODO

<!--
填写说明（发布前删除）：
1. 实验按「超参 → 架构 → 数据 → Bug Fix」逐级叠加，每一阶段以前一阶段的最优配置为起点。
2. 每个实验小节统一使用以下模板：
   - 负责人
   - 动机：为什么做这个实验，想验证什么
   - 设置：变量、对照组、训练步数、评测协议
   - 结果：表格，至少包含 Clean / Random 两列
   - 观察：1~3 句结论
3. 所有数字需注明训练步数与评测协议。协议不一致的结果不要放在同一张表中直接比较。
4. 旧版草稿见 blog_v1_draft.md，其中已有的数字和分析可以直接搬过来。
-->

---

## 摘要

TODO

---

## 主要结果：从 Baseline 到最终方案

<!-- TODO: 累积改进路线图（柱状图或折线图），横轴为各阶段，纵轴为 Clean / Random -->

| 阶段 | 累积改动 | Clean | Random | Avg |
|---|---|---|---|---|
| Baseline | - | | | |
| + 超参 | 见 §4 | | | |
| + 架构 | 见 §5 | | | |
| + 数据 | 见 §6 | | | |
| + Bug Fix | 见 §7 | | | |

---

## 1. 引言

TODO

- 背景：VLA 预训练 + 下游 SFT 范式
- 问题：仅用 clean 数据 SFT，在 randomized 上泛化差
- 本文：按「超参 → 架构 → 数据 → Bug Fix」系统梳理 SFT 后训练
- 贡献：TODO

---

## 2. 背景与实验设置

### 2.1 LingBot-VLA 2.0

TODO：模型结构简介（VLM、MoE 动作专家、Dual-Query 蒸馏、多视角输入）

LingBot-VLA 2.0 以 Qwen3-VL-4B 为 VLM 主干，三路相机图像各自经 ViT 编码成 64 个 token，与语言指令拼接。末尾再追加当前和未来两组可学习 query，整个 prefix 用 MRoPE 编码，内部是因果注意力。旁边并行一个同为 36 层的 Qwen2 结构动作专家，它的 FFN 全部换成 1 个共享专家加 32 个 sigmoid 路由专家（top-4）的 MoE，用 AdaRMSNorm 注入流匹配时间步。两条流每层拼接 QKV 做联合注意力：prefix 看不到动作，state 看不到动作，动作 token 之间双向互看并读取整个 prefix。这样 55 维统一动作空间里的动作块就靠流匹配去噪生成。训练时，两组 query 的输出经 Resampler 交叉注意力，分别回归冻结的 LingBot-Depth 深度特征和 DINO-Video 时序特征，对应当前帧和 T 步后的未来帧，让模型学会感知几何、预判场景变化。推理时不需要教师，prefix 只算一次存进 KV cache，之后约 10 步 Euler 去噪只跑动作专家。

### 2.2 任务与评测协议

TODO：训练数据、评测设置（clean / randomized）、每任务评测次数

训练数据使用 RoboTwin clean-50，直接使用 RoboTwin_lerobot_v30 数据集。（该数据集后续发现有残缺，下文会提到修复）

评测在 clean 和 randomized 分别测试 100 条，以抑制测试时波动。

### 2.3 训练配置

TODO：硬件、batch size、优化器、默认超参

训练使用 16\*H200 或 32\*H200；保证 global_batch_size 为 1024，micro_batch_size 为 32，不同卡数调整对应的 gradient_accumulation_num；优化器和其他超参与 LingBot-VLA 2.0 默认配置一致，除了 `学习率` 有修改，在下文中阐述。

---

## 3. Baseline

TODO：baseline 配置与结果，clean / randomized 的差距，典型失败案例

我们以 LingBot-VLA 2.0 官方仓库的 ecca77bb259b9592d5fc0eb2b4972d4a236ed2c8 commit 为 baseline，robot action 用 absolute joint，在 RoboTwin_lerobot_v30 的 clean-50 数据集上训练 20k step，作为 baseline。评测结果为 clean79.6、randomized33.6 这种情况下，几乎没有 c2r 能力。

---

<!-- ## 4. 超参

> 起点：Baseline（§3）

### 4.1 训练步数

负责人：TODO

### 4.2 学习率

负责人：TODO

### 4.3 动作空间

负责人：TODO

### 4.4 动作块执行长度

负责人：TODO

### 4.x TODO：其他超参实验

### 4.5 小结

TODO：本阶段最优配置，以及它作为下一阶段起点的配置说明 -->

## 4. 动作空间

我们尝试了四种动作空间：Absolute Joint、Relative Joint、Absolute EE、Relative EE。

|  | clean SR | randomized SR |
|---|---|---|
| `Absolute Joint` | 76.6 | 36.2 |
| `Relative Joint` | 77.0 | 34.7 |
| `Absolute EE` | 76.6 | 27.4 |
| `Relative EE` | 74.0 | 24.8 |

可以注意到 Absolute Joint > Relative Joint >> Absolute EE > Relative EE，因此我么最终选择了 Absolute Joint。这也是 LingBot-VLA 2.0 官方 c+r 进行 post train 时采用的配置。

---

## 4. 学习率

我们针对学习率的测试分为了两类：全局学习率、分模块学习率。

LingBot-VLA 2.0 官方给出的 lr 调度为 cosine 调度，从 1.0e-4 经过 50k step decay 到 5.0e-5，以其作为 baseline。由于我们在测试这组实验时，得到的结论为，训练到 30k step 就几乎没有提升，于是我们测试时考虑将 max_steps 设置为 30k。我们设计了两组 baselr 的实验。lr_1：从 1.0e-4 经过 30k step decay 到 1e-6；lr_2: 从 2e-5 经过 30k step decay 到 2e-7。实验结果如下：

|  | clean SR | randomized SR |
|---|---|---|
| `baseline` | 76.6 | 50.6 |
| `lr_1` | 78.4 | 50.6 |
| `lr_2` | - |  |

经过前面的测试，我们发现调小学习率并没有明显提升，因此我们没有修改 base_lr 而是选择加入分模块学习率。默认的配置下，LingBot-VLA 2.0 的全参数都是一套学习率，包括其中 VLM、LLM embedding等，这一点不太合理，因此我们采用了分模块的学习率。

|  | clean SR | randomized SR |
|---|---|---|
| `baseline` | 80.2 | 52.5 |
| `modlr` | 80.9 | 55.4 |

可以看到分模块学习率在 randomized 上有较大的提升。

## 4. 动作块执行长度

修改动作块执行长度，即推理大 chunk、执行少 step，这是常见的推理时优化策略。我们的 chunk_size 设置为 50，测试了 50、25、10 三组动作块执行长度

|  | clean SR | randomized SR |
|---|---|---|
| 50 | 79.8 | 47.5 |
| 25 | 78.5 | 45.8 |
| 10 | 68.5 | 23.6 |

从整体上看，随着动作块执行长度递减，SR也递减。这似乎与普遍认知中，“更精细的控制有更高的成功率“结论相违背。但是实际上，每个 task 都有其最佳的动作块执行长度，例如

动作块执行长度为 10 时最好：

| setting&task | ul=50 | ul=25 | ul=10 |
|---|---|---|---|
| clean&blocks_ranking_size | 43 | 61 | 80 |
| randomized&pick_dual_bottles | 60 | 70 | 84|
| randomized&pick_diverse_bottles | 53 | 50 | 67 |
| randomized&turn_switch | 8 | 15 | 22 |

动作块执行长度为 25 时最好：

| setting&task | ul=50 | ul=25 | ul=10 |
|---|---|---|---|
| clean&handover_block | 88 | 95 | 3 |
| randomized&put_bottles_dustbin | 51 | 63 | 5 |
| randomized&stack_blocks_two | 76 | 92 | 39 |
| randomized&blocks_ranking_size | 43 | 59 | 20 |

动作块执行长度为 50 时最好：

| setting&task | ul=50 | ul=25 | ul=10 |
|---|---|---|---|
| randomized&place_burger_fries | 97 | 92 | 17 |
| clean&open_microwave | 85 | 44 | 11 |
| randomized&place_container_plate| 88 | 75 | 16 |
| randomized&stack_bowls_two | 79 | 53 | 24 |

由此，每个 setting&task 都有其最佳的动作块执行长度，可惜官方规则不允许根据 setting&task 进行参数路由，因此我们没有深入测试。

---

## 5. 架构

> 起点：§4 最优配置

为探索如何将 SAM-mask 结合到 VLA，我们尝试了两类方案：将 SAM-mask 后的图作为新视角输入到 VLA 中、将 SAM-mask 后的图作为新的一个 MetaQuery 监督分支。

### 5.1 多视角输入（位置编码 / 注意力掩码）

对于作为新视角输入，我们将 SAM-mask 后的图作为第四个图传入 VLM。我们尝试了三种位置编码与两种attention mask。

| pos | 做法 | 对其他 token 的影响 |
|---|---|---|
| `native` | 4 张图一起走 `get_rope_index`，SAM 图就是普通的第 4 张图 | SAM 图占用新的位置段，语言 token 的 id 顺延 |
| `copy` | 先去掉 SAM 图按 3 图计算 id，再把 raw cam_high 的 (t, h, w) 整块复制给 SAM 图 | 其余 token 的 id 与 3 图模型完全一致；SAM 和 raw 在 RoPE 意义上"同位置" |
| `copy_t1` | 同 `copy`，但 SAM 块的 t 轴 +1（h、w 不变） | 同 `copy`；空间对齐，时间上错开一格，用来区分两个视角 |

| attn | 做法 | 用途 |
|---|---|---|
| `causal` | 不改 mask：SAM 能看到 raw，raw 看不到 SAM（SAM 排在后面） | 默认值，等价于"普通第 4 张图" |
| `pair` | 额外打开 raw → SAM，两个视角互相可见，各自内部仍 causal | round-1 消融 |

因此一共六组实验。

|  | clean SR | randomized SR |
|---|---|---|
| baseline | 76.6 | 36.2 |
| `native`, `causal` | 79.2 | 41.3 |
| `copy`, `causal` | 77.9 | 41.5 |
| `copy_t1`, `causal` | 78.5 | 42.6 |
| `native`, `pair` | 77.0 | 38.8 |
| `copy`, `pair` | 77.6 | 38.6 |
| `copy_t1`, `pair` | 77.5 | 37.6 |

可以明显得出结论，何种位置编码几乎不影响效果，但是 attention mask 有影响，主要在于，attention mask 应当与其 VLM 预训练时的 mask 方式一致。


### 5.2 MetaQuery

负责人：TODO

这组实验中，我们不修改输入到 VLM 的图片，而是在 LingBot-VLA 2.0 原有的基于 MetaQuery 的 Video 监督和 Depth 监督前再增加一组 Learnable Query，在 causal mask 下读取前面的 vlm token 并监督重建 SAM-mask 图。

|  | clean SR | randomized SR |
|---|---|---|
| baseline | 76.6 | 36.2 |
| meta query | 78.3 | 37.9 |


### 5.x TODO：其他架构实验

### 5.3 小结

TODO：本阶段最优配置

---

## 6. 数据

> 起点：§5 最优配置

### 6.1 Qwen-SAM 前景掩码 v1

负责人：TODO

我们最初通过 prompt engineering 让 Qwen3.8-27B 根据 RoboTwin task instruction 以及首帧图像，得出当前视角下的关键物品的名称，然后将其与第三视角图像送入 SAM，得到第三视角下的、目标物体分割mask，并将其应用在第三视角图像中，将mask外的部分降低亮度（像素值乘0.1），并在mask边缘打上高亮（线宽2像素），将处理后的第三视角图像替换原始的第三视角图像，以此来提示 VLA 的执行。

|  | clean SR | randomized SR |
|---|---|---|
| baseline | 76.6 | 36.2 |
| Qwen-SAM v1 | 79.8 | 47.5 |

### 6.2 Qwen-SAM 前景掩码 v2（VLM 复核）

负责人：TODO

我们测试发现 Qwen 虽然无法直接输出 mask，但是对于给定的图像与 mask 有着较强的分辨能力，因此我们考虑在 Qwen-SAM v1 的基础上，放宽松提示词模板，以获得更高的召回率，与此同时，将得到的 mask 连同图像与一个是非问题（“XX位置是XX物体吗”）再次送入 Qwen 以判断某个 mask 是否正确。通过这样的方式，我们实现了既提高了召回率又降低了错误率。

|  | clean SR | randomized SR |
|---|---|---|
| baseline | 76.6 | 36.2 |
| Qwen-SAM v1 | 79.8 | 47.5 |
| Qwen-SAM v2 | 80.2 | 49.8 |

### 6.3 颜色扰动

负责人：TODO

### 6.x TODO：其他数据实验

### 6.4 小结

TODO：本阶段最优配置

---

## 7. Depth / Video 蒸馏目标误用掩码图

负责人：TODO

在我们使用 SAM-mask 后的图替换第三视角输入后，最初我们没有考虑对 LingBot-VLA 2.0 自带的 Depth / Video 蒸馏分支的影响，即我们直接将 SAM-mask 后的图送入 Depth / Video 蒸馏分支的 Teacher 模型，而这么做显然会导致 Teacher 分布失真，因此我们修复了这个问题，在训练时，第三视角使用 SAM-mask 图，而在计算 Teacher 分布时使用原始第三视角图。

|  | clean SR | randomized SR |
|---|---|---|
| before fix | 80.2 | 49.8 |
| after fix | 80.2 | 52.5 |

虽然提升不大，但确实是一个值得修复的问题。

## 7. Depth / Video 蒸馏目标误用掩码图

负责人：TODO

在我们使用 SAM-mask 后的图替换第三视角输入后，最初我们没有考虑对 LingBot-VLA 2.0 自带的 Depth / Video 蒸馏分支的影响，即我们直接将 SAM-mask 后的图送入 Depth / Video 蒸馏分支的 Teacher 模型，而这么做显然会导致 Teacher 分布失真，因此我们修复了这个问题，在训练时，第三视角使用 SAM-mask 图，而在计算 Teacher 分布时使用原始第三视角图。

|  | clean SR | randomized SR |
|---|---|---|
| before fix | 80.2 | 49.8 |
| after fix | 80.2 | 51.5 |

虽然提升不大，但确实是一个值得修复的问题。

## 7. action_padding 与 lerobotv3.0 数据集修复

负责人：TODO

在最后的测试中，我们发现 click 相关任务成功率极低，因此考虑查看数据集相关的问题。我们发现了两点：

1. action_padding。 LingBot-VLA 2.0 默认逻辑为：在 episode 末尾的 chunk，复制最后的 action 填满 chunk，在计算 loss 时计算所有位置的 loss，即保证了 episode 结束后保持动作。但是这种方式，在没有 history memory 的 LingBot-VLA 2.0 中，会导致例如 click 相关的 task 无法区分 “即将 click” 与 “完成 click” 的状态，因此导致了 click 相关任务始终无法完成最后一步。于是我们考虑将逻辑修改为：对 chunk 进行 padding 但是 padding 的部分不计入 loss。这样让模型更有可能输出完成 click 动作的 chunk。

2. lerobotv3.0 数据集。RoboTwin官方 huggingface 给出的 lerobotv3.0 数据集存在 episode 前后错位情况，即部分 episode 中可能在末尾帧会读取到下一个 episode frame。这个问题仅存在于 lerobotv3.0 数据集，而 lerobotv2.1 数据集中一切正常。由于我们一直使用的 lerobotv3.0 数据集，因此选择通过 lerobotv2.1 数据集对其进行修复。

由于时间限制，上述两点修复没有独立消融效果。其整体提升效果如下：

|  | clean SR | randomized SR |
|---|---|---|
| before fix | 80.9 | 55.4 |
| after fix | 81.9 | 58.2 |

修复后也就是我们的最终版本。

### 7.x TODO：其他 bug

### 7.2 小结

TODO

---

## 8. 讨论

TODO：综合 §4~§7 的发现，讨论 SFT 对预训练表征的保留与破坏。每个阶段带来的增益，哪些改动与预训练一致，哪些偏离了预训练

---

## 9. 局限与未来工作

TODO

---

## 附录

### A. 逐任务结果

TODO

### B. Prompt 模板

TODO

### C. 完整训练超参

TODO
