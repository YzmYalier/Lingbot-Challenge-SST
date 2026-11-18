#!/usr/bin/env python3
"""Rebuild the standalone report from local Git history and benchmark statistics.

Run: python3 WebPage/source/build_report.py
OpenCV extracts qualitative video frames. No network requests or changes to experiment repositories.
"""
from pathlib import Path
import csv
import html
import json
import re
import subprocess

from draw_diagrams import build_diagrams
from draw_sam_examples import build_sam_examples
from i18n import bilingual_html

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
ROOT = SITE.parent
REPO = ROOT / 'lingbot-vla-v2'
BENCH = ROOT / 'VLABenchmarkResult'

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()

def audit():
    runs = []
    pattern = re.compile(r'^([\w.-]+)\s+\S+\s+(YES|NO|SKIP)\s+(\d+)/(\d+)\s+([\d.]+)%', re.M)
    for directory in sorted(BENCH.iterdir()):
        if not directory.is_dir():
            continue
        files = sorted(directory.glob('stats*.txt'))
        rows, headers, conflicts = {}, [], []
        for file in files:
            text = file.read_text(errors='replace')
            rel = str(file.relative_to(ROOT))
            headers.append({'file': rel, 'text': text.split('Task                           ')[0]})
            for task, done, success, total, _ in pattern.findall(text):
                row = {'task': task, 'done': done, 'success': int(success), 'total': int(total), 'source': rel}
                if task in rows and (rows[task]['success'], rows[task]['total']) != (int(success), int(total)):
                    conflicts.append(task)
                if task not in rows or file.name == 'stats.txt':
                    rows[task] = row
        if not rows:
            continue
        success = sum(x['success'] for x in rows.values())
        total = sum(x['total'] for x in rows.values())
        episodes = sorted({x['total'] for x in rows.values()})
        complete = len(rows) == 50 and len(episodes) == 1 and all(x['done'] == 'YES' for x in rows.values()) and not conflicts
        runs.append(dict(id=directory.name, tasks=len(rows), episodes=episodes, success=success, total=total,
                         rate=success / total * 100 if total else None, complete=complete, conflicts=conflicts,
                         files=[str(x.relative_to(ROOT)) for x in files], headers=headers, rows=list(rows.values())))
    return runs

RUNS = audit()
BY_ID = {x['id']: x for x in RUNS}
USED = {}

def select(model, step, scene, suffix=None, episodes=50):
    prefix = f'{model}_{step}_demo_{scene}_'
    found = [r for r in RUNS if r['id'].startswith(prefix) and r['complete'] and r['episodes'] == [episodes]]
    if suffix:
        found = [r for r in found if r['id'].endswith(suffix)]
    if len(found) != 1:
        raise ValueError(f'Ambiguous/missing selection: {prefix}, {suffix}, {episodes}: {[x["id"] for x in found]}')
    r = found[0]
    USED[r['id']] = r
    return r

BASE = 'robotwin_v30_formal_abs'
SAM = BASE + '_sam_multi'
JUDGE = BASE + '_sam_judge_0.1_multi'
RAW = JUDGE + '_rawteacher'
MOD = RAW + '_modlr'
FIX = 'robotwin_v30_modlr_no_wristaug_datafix'

def pair(model, step, c=None, r=None, episodes=50, clean=True, random=True):
    return dict(step=step, episodes=episodes,
                clean=select(model, step, 'clean', c, episodes) if clean else None,
                random=select(model, step, 'randomized', r, episodes) if random else None)

P = {
 'n0': pair(BASE, 15000, episodes=100),
 'n1': pair(SAM, 15000, '20261006_150800', '20261005_20000'),
 'n2': pair(JUDGE, 15000),
 'n3': pair(RAW, 15000),
 'n4': pair(MOD, 15000),
 'n5': pair(FIX, 30000, episodes=100),
}

def score(r):
    if r is None:
        return '<span class="missing" title="该项未报告">—</span>'
    return f'<strong class="score">{r["rate"]:.1f}</strong>'

def table(rows, caption='成功率 SR（%）', reference=None):
    out = [f'<div class="table-scroll" tabindex="0" role="region" aria-label="{html.escape(caption)}"><table><thead><tr><th scope="col">实验配置</th><th scope="col">Step</th><th scope="col">Ep / 任务</th><th scope="col">Clean (%) ↑</th><th scope="col">Random (%) ↑</th></tr></thead><tbody>']
    for label, p in rows:
        out.append(f'<tr><th scope="row">{label}</th><td>{p["step"] // 1000}k</td><td>{p["episodes"]}</td><td>{score(p["clean"])}</td><td>{score(p["random"])}</td></tr>')
    return ''.join(out) + '</tbody></table></div>'

def mini_table(headers, rows, caption):
    return '<div class="table-scroll" tabindex="0" role="region" aria-label="'+html.escape(caption)+'"><table><thead><tr>'+''.join('<th scope="col">'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(('<th scope="row">'+x+'</th>') if i==0 else ('<td>'+x+'</td>') for i,x in enumerate(row))+'</tr>' for row in rows)+'</tbody></table></div>'

DIAGRAMS = build_diagrams()
DIAGRAMS['sam_examples'] = build_sam_examples()

def diagram(name):
    labels = {'sam_examples': '原图与 SAM 前景输入对照', 'extra': '03.a 视觉输入结构', 'mask': '03.a Attention mask 对照', 'query': '03.b MetaQuery 结构'}
    return '<figure class="method-figure"><div class="diagram-scroll" tabindex="0" role="region" aria-label="'+labels[name]+'">'+DIAGRAMS[name]+'</div></figure>'

MODULES = []
def module(id, number, title, kind, parent, change, content, takeaway):
    MODULES.append(dict(id=id, number=number, title=title, kind=kind, parent=parent, change=change, content=content, takeaway=takeaway))

module('n0','01','基线：Clean 与 Randomized 的差距','main','LingBot-VLA 2.0 预训练模型',
 '使用 RoboTwin clean-50 做监督微调，动作表示为 absolute joint。',
 '<p>我们首先沿用 LingBot-VLA 2.0 的后训练配方，在 clean 数据上微调，并分别在 Clean 与 Randomized 场景中评测。模型在熟悉的场景中已经能完成大多数任务，但背景、光照和干扰物发生变化后，成功率明显下降。</p>'+
 table([('Absolute joint',P['n0']),('Absolute joint',pair(BASE,20000,'20261008_004609','20261008_003424'))])+
 '<p>15k 时，Clean 成功率为 77.1%，Randomized 只有 36.2%。继续训练到 20k，Clean 提高到 79.6%，Randomized 却没有改善。这让我们把重点转向输入与监督方式：如何让模型学到的操作能力少受场景外观影响？</p>',
 '仅延长 clean 数据上的训练，无法解决我们观察到的泛化差距。')

module('b-action','01.a','动作空间：joint 优于 end-effector','branch','N0 · 原始三视角输入',
 '比较 absolute / relative 与 joint / end-effector 两个维度。',
 '<p>在改变视觉输入之前，我们先检查动作表示是否限制了表现。四组实验采用相同训练步数和评测次数。</p>'+
 table([('Absolute joint',P['n0']),('Relative joint',pair('robotwin_v30_formal_rel',15000,episodes=100)),('Absolute EE',pair(BASE+'_ee',15000,episodes=100)),('Relative EE',pair('robotwin_v30_formal_rel_ee',15000,episodes=100))])+
 '<p>Joint 表示在 Randomized 上明显优于 EE；absolute 与 relative joint 则十分接近。我们因此沿用 absolute joint，把后续实验集中在视觉泛化上。</p>',
 '后续统一采用 absolute joint。')

module('n1','02','前景掩码：压暗背景，突出操作对象','main','N0 · 原始三视角输入',
 '用 Qwen + SAM3 生成任务相关前景，替换主视角图像。',
 '<p>我们的第一个假设是：clean 与 randomized 的外观差异会分散模型对操作对象的注意力。如果先突出任务相关前景，模型就可能更容易复用已经学到的操作能力。</p><p>具体来说，Qwen 根据任务指令和图像提出目标物体，SAM3 生成分割掩码。我们保留前景，把背景像素乘以 0.2，并用处理后的图像替换主视角 cam_high；两个腕部视角保持不变。训练和推理都使用相应的前景处理。</p><p>我们还比较了两种 episode–instruction 配对方式。<strong>Single</strong> 沿用默认 LeRobot 数据集中 episode 与 instruction 的一一配对关系，为每个 episode 按其对应指令生成 SAM mask。</p><p><strong>Multi</strong> 则对每个 episode 分别使用 <strong>100 条 seen instructions</strong> 生成 SAM mask，得到与各条指令对应的 SAM 图像序列。训练时随机选择其中一组 SAM 图像–instruction 配对，将当前帧的 SAM 图像与对应指令一起送入 VLA。</p>'+
 table([('原图 · N0',P['n0']),('SAM single',pair(BASE+'_sam_single',15000,'20261005_124200','20261005_124200')),('SAM multi · N1',P['n1']),('SAM multi',pair(SAM,20000,'20261006_122200','20261006_122200'))])+
 '<p>在 15k 时，multi 将 Randomized 从 36.2% 提高到 47.8%，提升 <strong>11.6 个百分点</strong>，Clean 也略有提高。这是整轮实验中最明显的一步。Multi 相比 single 再提高 2.7 个百分点，我们在后续实验中继续随机采样配对的 SAM 图像与 instruction。</p>',
 '直接处理主视角输入，就能明显改善随机场景表现。')

CHUNK_RUNS = {
 50: pair(SAM,20000,'20261006_122200','20261006_122200'),
 25: pair(SAM,20000,'ul25_20261007_181600','ul25_20261007_154000'),
 10: pair(SAM,20000,'ul10_20261007_154500','ul10_20261007_155000'),
}
chunk_task_rows = []
for scene,task in [
    ('clean','blocks_ranking_size'),
    ('random','pick_dual_bottles'),
    ('random','blocks_ranking_size'),
    ('random','stack_blocks_two'),
    ('clean','open_microwave'),
    ('random','place_container_plate'),
]:
    rates = []
    for length in (50,25,10):
        row = next(r for r in CHUNK_RUNS[length][scene]['rows'] if r['task'] == task)
        rates.append(100 * row['success'] / row['total'])
    best = max(rates)
    cells = [f'<strong class="score">{rate:.0f}</strong>' if rate == best else f'{rate:.0f}' for rate in rates]
    winners = ' / '.join(str(length) for length,rate in zip((50,25,10),rates) if rate == best)
    chunk_task_rows.append([task, 'Clean' if scene == 'clean' else 'Randomized', *cells, winners+' 步'])

module('b-chunk','02.a','动作块：最优执行长度因任务而异','branch','N1 · 同一个 20k 模型',
 '预测长度保持 50 步，每轮分别执行前 50、25 或 10 步。',
 '<p>更频繁地观察和重规划，是否能提高闭环控制的成功率？我们固定模型权重，只调整动作块的实际执行长度。</p>'+
 table([(f'执行 {length} 步',CHUNK_RUNS[length]) for length in (50,25,10)])+
 '<p>从整体平均看，执行 25 步时表现略降；缩短到 10 步后，Randomized 下降到 23.4%。但这一趋势并不适用于所有任务。<strong>每个 task 都有适合自己的最优执行长度，同一任务的最优值还可能随场景改变。</strong></p><h4>逐任务比较</h4><p>以下列出部分任务在三种执行长度下的成功率，每个设置评测 50 次，粗体为本行最高值。</p>'+
 mini_table(['Task','场景','50 步 (%)','25 步 (%)','10 步 (%)','本组最优'],chunk_task_rows,'不同动作块执行长度的逐任务成功率')+
 '<p>例如，blocks_ranking_size 在 Clean 下更适合执行 10 步，成功率从 50 步时的 42% 提高到 80%；换到 Randomized 后，最优变为 25 步，达到 58%，而 10 步只有 20%。pick_dual_bottles 在 Randomized 下也受益于更短的执行长度，从 50 步的 60% 提高到 10 步的 84%。</p><p>另一方面，open_microwave 在 Clean 下执行 50 步达到 84%，缩短到 10 步后仅为 8%。这些结果说明，更频繁地重规划并不总能提高成功率，需要结合任务及场景选择执行长度。在本次测试的 10、25、50 步中，若所有任务统一使用一套配置，50 步的整体表现仍然最好，因此后续主线实验沿用 50 步。</p>',
 '每个 task 都有适合自己的最优执行长度，最优值还受场景影响；统一配置下，本组实验仍以 50 步的整体表现最好。')

module('n2','03','掩码复核：提高前景质量，保留少量背景','main','N1 · SAM multi',
 '增加 VLM 掩码复核，并将背景亮度设为 ×0.1、边缘描边设为 2px。',
 '<p>初版掩码仍会漏掉相关物体，或包含错误区域。我们放宽候选物体的召回，再把图像、候选掩码和物体名称送回 Qwen 复核，筛除错误掩码。同时，我们将背景亮度降到原图的 0.1，并加入 2px 描边。</p>'+diagram('sam_examples')+
 table([('SAM multi · N1',P['n1']),('复核 + 背景 ×0.1 · N2',P['n2']),('复核 + 背景全黑',pair(BASE+'_sam_judge_0_multi',15000))])+
 '<p>更新后，Randomized 从 47.8% 提高到 50.8%，Clean 从 78.0% 降到 76.0%。我们还尝试了完全涂黑背景，结果两个场景都变差。这说明前景之外的上下文仍然有用，适度压暗比彻底移除更适合当前模型。</p><p>这一组同时更新了掩码复核和渲染方式，+3.0 pp 对应两者的联合改动。</p>',
 '保留复核后的多掩码输入、0.1 倍背景亮度与描边。')

extra_rows=[('原图 · N0',P['n0']),('替换主视角 · N2',P['n2'])]
for pos,att in [('native','causal'),('copy','causal'),('copy_t1','causal'),('native','pair'),('copy','pair'),('copy_t1','pair')]:
    extra_rows.append((f'{pos} / {att}',pair(BASE+f'_sam_extra_view_judge_0.1_multi.pos-{pos}.attn-{att}.align-raw',15000,'20261012_190000','20261012_190000',clean=(pos,att)!=('copy_t1','pair'))))
module('b-extra','03.a','第四视角：同时看原图与 SAM 图','branch','N0 · 原始三视角输入',
 '保留原始主视角，将 SAM 图加入第四个图像槽位。',
 '<p>我们从 N0 的原始三视角输入出发，保留原始主视角，再增加一张 SAM 前景图。与 N2 直接替换主视角相比，这种方式让模型同时看到原图与前景信息。因此，我们同时比较原图基线 N0 和输入替换方案 N2。</p>'+diagram('extra')+'<p>我们比较三种位置编码：native 为第四张图分配新位置；copy 复用主视角的 (t,h,w)；copy_t1 在复用空间位置的同时把时间位置加一。注意力分别使用原有 causal mask，以及 pair mask。后者只将“原图 Query → SAM Key”整块设为可见，各视角内部仍为下三角。</p>'+diagram('mask')+table(extra_rows)+
 '<p>保留 causal attention 时，三种位置编码的 Randomized 成功率为 <strong>41.3%–42.6%</strong>，相对 N0 的 36.2% 提高 5.1–6.4 个百分点，说明从输入侧增加 SAM 信息确实有帮助。不过，它们仍低于 N2 的 50.8%：在当前任务中，直接弱化原图中的背景干扰，比保留原图并追加前景提示更有效。</p><p><strong>位置编码方式不是这组实验的主要影响因素。</strong>在 causal 设置下，三种位置方案仅相差约 1.3 个百分点，复用空间位置或额外区分时间位置都没有带来明显优势。相比之下，同一种位置编码从 causal 改成 pair 后，成功率分别下降 2.6、3.2 和 5.0 个百分点，方向一致。</p><p>新增信息的接入应优先保持 VLM 原生的注意力形式。Causal 保留了预训练时的 token 可见关系；pair 虽然增加了原图与 SAM 图之间的信息交换，却同时改变了这种交互结构，结果反而更差。与设计复杂的位置编码相比，贴合原生 attention 更值得优先考虑。</p>',
 '增加视觉输入时，保持 VLM 原生的 causal attention 比选择何种位置编码更重要；直接替换主视角仍是本组最有效的方案。')

module('b-query','03.b','MetaQuery：把掩码作为辅助监督','branch','N0 · 原始三视角输入',
 '在 depth/video query 前增加一组可学习 query，预测主视角掩码。',
 '<p>这组实验同样从 N0 出发，保持全部原图输入，把 SAM 信息从后端的 MetaQuery 监督引入。我们增加一组可学习 query，让它在因果注意力下读取图像和语言，再通过 MaskFormer 式点积预测复核后的掩码。动作专家也能读取这组 query。下表同时列出 N0 和 N2，以比较辅助监督与直接处理输入的效果。</p>'+diagram('query')+table([('原图 · N0',P['n0']),('替换主视角 · N2',P['n2']),('SAM MetaQuery',pair(BASE+'_sam_query_judge_multi_B',15000,clean=False))])+
 '<p>MetaQuery 的 Randomized 成功率为 <strong>37.9%</strong>，比 N0 高 1.7 个百分点，但比 N2 低 12.9 个百分点。与 <a href="#b-extra">03.a</a> 保留 causal attention 的第四视角方案相比，它也低了 3.4–4.7 个百分点。</p><p>这两类方案都希望为 VLA 增加前景信息，区别在于接入位置：输入侧方案在推理时直接提供 SAM 图，让 VLM 读取额外视觉信息；MetaQuery 则依靠训练时的掩码重建目标，引导模型形成辅助表征。后者引入了额外的训练目标，但这份辅助监督并未转化为同等幅度的操作成功率提升。</p><p>综合两组实验，我们更倾向于<strong>直接从输入侧接入新信息，并尽量保留 VLM 原生的 attention 形式</strong>。在本组设置下，直接替换主视角最好，保留 causal attention 的第四视角次之，后端 MetaQuery 监督收益较小。输入侧方案的代价是推理时也需要生成 SAM 图。</p>',
 '前景信息从输入侧接入，比通过后端 MetaQuery 辅助监督引入更有效；这一优势在保持原生 causal attention 时最清楚。')

module('n3','04','蒸馏目标：让 teacher 继续读取原图','main','N2 · 复核后的 SAM 主视角输入',
 'policy 读取 SAM 图，depth/video teacher 读取同步原图，并对前景区域加权。',
 '<p>替换主视角之后，我们重新检查了模型自带的 depth/video 蒸馏。原先 policy 与 teacher 共用处理后的图像，但压暗和描边也会改变 teacher 的输入分布。我们因此为 teacher 单独提供同步原图，policy 仍然读取 SAM 图。</p><p>Raw 与 masked 图共享同一组增强参数；对齐损失按每个 patch 的前景比例加权，让监督更集中在任务相关区域。</p><div class="method-flow"><div><strong>SAM 主视角 + 腕部双视角</strong></div><span class="flow-symbol">↔</span><div><strong>原图 → Depth / Video teacher</strong></div></div>'+table([('Masked teacher · N2',P['n2']),('Raw teacher + 前景加权 · N3',P['n3']),('背景全黑 · masked teacher',pair(BASE+'_sam_judge_0_multi',15000)),('背景全黑 · raw teacher',pair(BASE+'_sam_judge_0_multi_rawteacher',15000))])+
 '<p>背景 ×0.1 与全黑两种设置下，Randomized 分别提高 2.3 和 3.9 个百分点。两个方向一致的结果，让我们保留了“原图 teacher + 前景加权”的联合方案。</p>',
 '为 policy 和 teacher 分别选择输入，可以保留前景处理的收益，同时改善蒸馏监督；推理时不增加 teacher 开销。')

module('b-depth','04.a','深度视角：显式几何输入是否有帮助','branch','N3 · SAM 主视角与原图 teacher',
 '追加深度伪彩色图，共享 ViT，使用独立 depth merger。',
 '<p>恢复 teacher 输入后，我们进一步尝试把几何信息显式送入策略。我们将 LingBot-Depth 的深度输出转换成 inverse-depth 伪彩色图，作为第四路输入，并为它设置独立的连接器，从 RGB merger 初始化。</p>'+table([('三视角方案 · N3',P['n3']),('Depth map + separate connector',pair(RAW+'_depth_map_sepconn',15000)),('Depth map + separate connector',pair(RAW+'_depth_map_sepconn',20000))])+
 '<p>15k 时，深度方案的 Clean 略高，但 Randomized 只有 48.2%，比三视角方案低约 5 个百分点。到 20k 时也没有超过对应的三视角结果。额外的几何输入在当前训练预算下没有带来预期收益。</p><p>一种可能的解释是，Depth 图本身并不适合作为独立图像，经由 Visual Encoder 编码后送入 LLM。深度信息或许更适合用于对齐监督，让模型在训练中学习几何结构。</p>',
 '最终保留三视角输入，几何信息继续通过蒸馏目标参与训练。')

module('b-lr','04.b','全局学习率：统一调小没有获益','branch','N3 · 原图 teacher 与前景加权',
 '将调度周期缩短为 30k，比较更快衰减与更小初始学习率。',
 '<p>输入处理之外，我们也考察优化过程。LingBot-VLA 2.0 的默认学习率在 50k 内从 1e−4 衰减到 5e−5。考虑到这一学习率在 post-training 场景中偏高，我们主要尝试降低学习率：一是在 30k 内衰减到更小的终点，二是把初始学习率一起降低。</p>'+
 mini_table(['配置','初始 LR','最低 LR','调度终点'],[['默认','1e−4','5e−5','50k'],['lr_1','1e−4','1e−6','30k'],['lr_2','2e−5','2e−7','30k']],'Cosine 学习率设置')+
 table([('默认 · N3',P['n3']),('lr_1',pair(RAW+'_lr1',15000)),('lr_2',pair(RAW+'_lr2',15000,clean=False)),('默认 · N3',pair(RAW,30000)),('lr_1',pair(RAW+'_lr1',30000))])+
 '<p>更快衰减的 lr_1 在 15k 和 30k 都低于默认配置；初始学习率更小的 lr_2 在 15k 时只有 19.5%。统一降低所有参数的更新幅度，至少在当前训练预算下并不合适。</p>',
 '保留基础学习率调度，进一步按模块控制更新幅度。')

module('n4','05','分模块学习率：减少预训练表征的更新','main','N3 · 原图 teacher 与前景加权',
 '降低 ViT、visual merger 和 LLM 的学习率，冻结 token embedding。',
 '<p>视觉、语言与动作模块承担不同角色，也具有不同的预训练基础。我们希望视觉语言模块保留更多已有表征，同时让动作与对齐模块继续适应下游任务，因此为不同模块设置学习率倍率。</p>'+
 mini_table(['模块','学习率倍率'],[['ViT','0.3 ×'],['Visual merger','0.5 ×'],['LLM','0.5 ×'],['Token embedding','0 · 冻结'],['Action expert / Action I/O / Align','1.0 ×']],'相对于基础学习率的倍率')+
 '<p>基础 cosine 调度保持不变，MoE routed experts 原有的额外缩放也继续保留。</p>'+
 table([('统一 LR · N3',P['n3']),('分模块 LR · N4',P['n4']),('统一 LR · N3',pair(RAW,25000)),('分模块 LR · N4',pair(MOD,25000)),('分模块 LR · N4',pair(MOD,35000))])+
 '<p>15k 和 25k 时，Randomized 都提高了约 2.2 个百分点。与统一调小学习率相比，按模块分配更新幅度更符合我们对预训练表征保留与下游适应的需求。</p>',
 '分模块学习率进入最终方案。')

module('b-wrist','05.a','腕部增强：颜色扰动没有稳定收益','branch','N4 · 分模块学习率',
 '只对两个腕部相机施加训练时的光度与颜色增强。',
 '<p>主视角已经经过前景处理，剩余的外观差异可能来自腕部相机。我们根据随机场景的光照统计设置增强范围，加入亮度、对比度、饱和度、色调、gamma 与噪声等扰动，两个腕部相机共享一次光照采样。</p>'+table([('无腕部增强 · N4',P['n4']),('Wrist augmentation',pair(MOD+'_wristaug',15000,'20261023_140142','20261023_205936')),('无腕部增强 · N4',pair(MOD,25000)),('Wrist augmentation',pair(MOD+'_wristaug',25000,'20261024_041048','20261024_111836'))])+
 '<p>15k 和 25k 的 Randomized 都没有改善；20k 时则从 53.8% 提高到 55.8%。不同训练节点的结果方向不一致，Clean 也有下降，尚未观察到明确、稳定的增益。</p><p>测试进行到这里时，已经临近初赛结束，因此我们选择关闭腕部增强。当然，也不能排除训练仍不充分，尚未发挥出增强的作用。</p>',
 '最终方案关闭腕部颜色增强。')

module('b-task','05.b','任务采样：增加困难任务的训练机会','branch','N4 · 从 25k 模型继续训练',
 '根据逐任务失败率分配采样概率，在任务内部不放回抽帧。',
 '<p>部分任务长期落后于整体平均。我们尝试提高这些任务的采样概率：用 25k 模型在两个场景下的平均失败率，加上 ε=0.05 构造权重，训练时先抽任务，再抽帧。</p>'+table([('普通采样 · N4',pair(MOD,25000)),('任务加权采样',pair(MOD+'_tasksample_from25k',30000,'20261022_230633','20261022_225340')),('普通采样 · N4',pair(MOD,35000)),('任务加权采样',pair(MOD+'_tasksample_from25k',35000,'20261023_181601','20261023_180907'))])+
 '<p>从普通采样的 N4 25k 模型出发，任务加权采样续训到 30k 后，Randomized 从 55.2% 提高到 56.7%，有约 <strong>1.5 个百分点</strong>的明显提升；Clean 则从 80.6% 略降到 80.4%。不过，缺少 30k 的普通采样对照，尚不能将这一提升完全归因于采样方式。</p><p>到 35k 时，普通采样的 Randomized 为 56.2%，反而优于任务加权采样的 54.6%，Clean 也更高。这可能说明任务加权采样在从 25k 继续训练 10k 后已经过度训练，也可能说明这套采样策略本身无效。由于时间有限，且比赛官方明确禁止这类基于评测反馈调整任务采样的做法，我们没有进行进一步实验。</p>',
 '最终沿用普通采样。')

module('n5','06','数据修复：排除 padding，恢复帧对齐','main','N4 · 分模块学习率，不使用腕部增强与任务加权采样',
 '排除补齐动作的 loss，修复视频与动作轨迹的时间对齐，并重新生成 SAM 输入。',
 '<p>Click 类任务的表现不佳，促使我们回头检查训练监督。在这个过程中，我们发现了两类问题。</p><h4>Padding action 参与了监督</h4><p>Episode 末尾不足 50 步时，最后一个动作会被重复填充。原有 loss 也监督这些补齐位置，可能过度强化末端的保持动作。我们使用 padding mask，在 loss 的分子和归一化分母中同时排除这些位置，并与有效关节 mask 组合。</p><h4>视频与动作轨迹存在错位</h4><p>部分视频片段短于对应的数值轨迹，取帧可能越过 episode 边界，或与动作发生错位。我们重建了帧对齐的训练数据，重新生成 SAM 输入，并增加取帧边界检查。</p>'+table([('修复前 · N4',pair(MOD,20000)),('联合修复 · N5',pair(FIX,20000)),('联合修复 · N5',pair(FIX,25000,episodes=100)),('最终模型 · N5',P['n5']),('联合修复 · N5',pair(FIX,40000))])+
 '<p>同为 20k、每任务 50 次评测时，Randomized 从 53.8% 提高到 57.0%，提升 <strong>3.2 个百分点</strong>；Clean 从 80.9% 降到 78.9%。继续训练到 30k 后，最终模型达到 Clean <strong>81.86%</strong>、Randomized <strong>58.22%</strong>，平均成功率为 <strong>70.04%</strong>。</p><p>相对修复前的 N4，click_bell 与 click_alarmclock 的改善尤其明显。数据监督值得检查，而不仅仅是增加训练步数。</p><p>两项修复同时加入，尚未分别消融。下图比较<strong>修复前 N4（25k，分模块学习率）</strong>与<strong>联合修复后 N5（25k）</strong>，在每个场景、每个任务上分别评测 50 与 100 次。</p><div id="task-repair-figure"></div>',
 '前景处理、原图 teacher、分模块学习率与数据修复，共同组成我们的最终方案。')

def render_module(m):
    return f'''<section class="experiment {m['kind']}" id="{m['id']}" data-section>
      <div class="experiment-heading"><span class="section-number">{m['number']}</span><div><h3>{m['title']}</h3></div></div>
      <dl class="experiment-meta"><div class="wide"><dt>实验基线</dt><dd>{m['parent']}</dd></div><div class="wide"><dt>本次改动</dt><dd>{m['change']}</dd></div></dl>
      <div class="prose">{m['content']}</div><p class="takeaway"><span>结论</span>{m['takeaway']}</p></section>'''

def graph():
    # Reading map of experimental configurations, independent of code history.
    nodes=[
      ('n0','N0','Absolute joint 基线','原始三视角图像',20,15,'main',None),
      ('b-action','01.a','动作空间','Absolute / Relative · Joint / EE',350,15,'branch','n0'),
      ('b-extra','03.a','SAM 第四视角','保留原图，追加前景图',350,125,'branch','n0'),
      ('b-query','03.b','SAM MetaQuery','原始图像输入 + 掩码辅助监督',680,125,'branch','n0'),
      ('n1','N1','SAM 前景输入','背景 ×0.2 · 多掩码采样',20,235,'main','n0'),
      ('b-chunk','02.a','动作块执行长度','同一模型 · 执行 50 / 25 / 10 步',350,235,'branch','n1'),
      ('n2','N2','掩码复核与渲染','VLM 复核 · 背景 ×0.1 · 描边',20,345,'main','n1'),
      ('n3','N3','原图 teacher + 前景加权','Policy 看前景图，teacher 看原图',20,455,'main','n2'),
      ('b-depth','04.a','深度第四视角','深度伪彩色图 + 独立连接器',350,565,'branch','n3'),
      ('b-lr','04.b','全局学习率','更快衰减 / 更小初始学习率',680,565,'branch','n3'),
      ('n4','N4','分模块学习率','降低 ViT / LLM 学习率，冻结 embedding',20,675,'main','n3'),
      ('b-wrist','05.a','腕部颜色增强','只增强腕部双视角',350,785,'branch','n4'),
      ('b-task','05.b','任务加权采样','从 25k 模型继续训练',680,785,'branch','n4'),
      ('n5','N5','数据修复','Padding loss mask + 视频帧对齐',20,895,'main','n4'),
    ]
    outcomes={'b-action':'Joint 优于 EE','b-chunk':'因任务而异','b-extra':'未超过主线','b-query':'提升有限','b-depth':'未超过主线','b-lr':'没有收益','b-wrist':'收益不稳定','b-task':'收益不稳定'}
    lookup={n[0]:n for n in nodes}
    paths=[]; cards=[]
    for id,label,title,sub,x,y,kind,parent in nodes:
        if parent:
            p=lookup[parent];px,py=p[4],p[5]
            if y==py:
                d=f'M {px+292} {py+48} H {x}'
            elif x==px:
                d=f'M {x+26} {py+96} V {y}'
            else:
                mid=py+101
                d=f'M {px+26} {py+96} V {mid} H {x+26} V {y}'
            paths.append(f'<path class="edge {"main-edge" if kind=="main" else ""}" data-parent="{parent}" data-child="{id}" d="{d}"/>')
        result=f'{P[id]["random"]["rate"]:.1f}%' if id in P else outcomes[id]
        if id in P:
            label += f' · {P[id]["step"] // 1000}k'
        cards.append(f'<a href="#{id}" class="graph-node {kind}" style="left:{x}px;top:{y}px" data-node="{id}"><span class="node-top"><b>{label}</b><span>{result}</span></span><strong>{title}</strong></a>')
    return '<div class="graph-scroll" tabindex="0" role="region" aria-label="实验路线图"><div class="graph-canvas"><svg viewBox="0 0 1000 1010" aria-hidden="true">'+''.join(paths)+'</svg>'+''.join(cards)+'</div></div>'


def main():
    final=P['n5']; base=P['n0']
    # Register the 25k pairs for the task-level repair figure and appendix.
    before=pair(MOD,25000); after=pair(FIX,25000,episodes=100)
    tasks={}
    for name in ('clean','random'):
        for label,r in [('baseline',base[name]),('final',final[name]),('before',before[name]),('after',after[name])]:
            for row in r['rows']:
                tasks.setdefault(row['task'],{})[name+'_'+label]=row['success']/row['total']*100
    data={'mainline':[dict(id=k,label=next(m['title'] for m in MODULES if m['id']==k),step=p['step'],episodes=p['episodes'],clean=p['clean']['rate'],random=p['random']['rate']) for k,p in P.items()], 'tasks':[dict(name=k,**v) for k,v in sorted(tasks.items())]}
    git_graph=git('log','--all','--graph','--decorate','--oneline','origin/main..')
    # Retain internal provenance checks when rebuilding the results.
    expected={'a730cc1':'447044e','1d0d70e':'447044e','f85066c':'447044e','555fb09':'1d0d70e','597a83e':'1d0d70e','66ca08a':'c865be6','4f8512f':'d572309','217b4da':'dd049d0','6691b7e':'7b17403'}
    for child,parent in expected.items():
        assert git('rev-parse',child+'^')==git('rev-parse',parent),(child,parent)
    summary=table([(f'<a href="#{k}">{k.upper()}</a> · '+label,P[k]) for k,label in [('n0','Absolute joint'),('n1','SAM multi'),('n2','Judged SAM'),('n3','Raw teacher + weighted loss'),('n4','Module LR'),('n5','Data integrity fix')]], '主线实验 · N0–N4 为 15k，N5 为 30k')
    replacements={
      '@@STYLE@@':(HERE/'report.css').read_text(),
      '@@SCRIPT@@':(HERE/'report.js').read_text(),
      '@@WIDGET_SCRIPT@@':(HERE/'engagement.js').read_text(),
      '@@WIDGETS@@':json.dumps(json.loads((HERE/'widgets.json').read_text()),ensure_ascii=False).replace('</','<\\/'),
      '@@GRAPH@@':graph(),
      '@@MODULES@@':''.join(render_module(m) for m in MODULES),
      '@@SUMMARY@@':summary,
      '@@DATA@@':json.dumps(data,ensure_ascii=False).replace('</','<\\/'),
    }
    template=(HERE/'report.template.html').read_text()
    for key,val in replacements.items():
        template=template.replace(key,val)
    template=bilingual_html(template)
    for dest,root_prefix,site_prefix in [(SITE/'index.html','../',''),(ROOT/'index.html','','WebPage/')]:
        dest.write_text(template.replace('@@ROOT@@',root_prefix).replace('@@SITE@@',site_prefix))
    (SITE/'data'/'report-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (SITE/'data'/'evidence.json').write_text(json.dumps(list(USED.values()),ensure_ascii=False,indent=2))
    (SITE/'data'/'git-graph.txt').write_text(git_graph+'\n')
    with (SITE/'data'/'results.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['run','tasks','episodes_per_task','success','total','success_rate_percent','stats_files'])
        for r in USED.values():
            w.writerow([r['id'],r['tasks'],r['episodes'][0],r['success'],r['total'],f'{r["rate"]:.4f}',';'.join(r['files'])])
    print(f'Built two standalone HTML files; {len(MODULES)} modules, {len(USED)} verified runs, {len(tasks)} tasks.')

if __name__=='__main__':
    main()
