#!/usr/bin/env python3
"""Generate the report's editable, self-contained SVG method figures."""
from pathlib import Path
from html import escape

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
INK = '#273731'
MUTED = '#63746b'
LINE = '#ced8d0'
PAPER = '#fafbf8'
TEAL = '#356e65'
GREEN = '#eaf3ee'
VISIBLE = '#87aa9a'
COPPER = '#aa573b'
WARM = '#fbf0e8'
NEW = '#cd8c6c'
BLOCKED = '#f0f1ec'


class Drawing:
    def __init__(self, name, height, title, description):
        self.name, self.height = name, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" id="{name}" class="method-svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-labelledby="{name}-title {name}-desc">',
            f'<title id="{name}-title">{escape(title)}</title>',
            f'<desc id="{name}-desc">{escape(description)}</desc>',
            f'''<defs><style>
#{name} text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei', sans-serif; font-size: 16px; fill: {INK}; }}
#{name} .figure-title {{ font-size: 22px; font-weight: 600; }}
#{name} .strong {{ font-weight: 600; }}
#{name} .muted {{ fill: {MUTED}; }}
#{name} .teal {{ fill: {TEAL}; }}
#{name} .copper {{ fill: {COPPER}; }}
</style>''']
        for color, key in [(MUTED, 'neutral'), (TEAL, 'teal'), (COPPER, 'copper')]:
            self.parts.append(f'<marker id="{name}-arrow-{key}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M 0 0 L 7 3.5 L 0 7 Z" fill="{color}"/></marker>')
        self.parts.append('</defs>')
        self.rect(0, 0, 960, height, fill='white', stroke='none', radius=0)
        self.text(32, 40, title, kind='figure-title')

    def rect(self, x, y, w, h, fill=PAPER, stroke=LINE, radius=6, dash=False, width=1):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')

    def text(self, x, y, value, kind='', anchor='start', rotate=None):
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ''
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="middle" class="{kind}"{transform}>{escape(value)}</text>')

    def path(self, d, color=MUTED, arrow=False, dash=False, width=1.5):
        key = 'teal' if color == TEAL else 'copper' if color == COPPER else 'neutral'
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'+(f' marker-end="url(#{self.name}-arrow-{key})"' if arrow else '')+(' stroke-dasharray="5 5"' if dash else '')+'/>')

    def box(self, x, y, w, h, lines, accent=False, training=False):
        fill, stroke = (WARM, '#d9bca9') if training else (GREEN, '#9cbbac') if accent else (PAPER, LINE)
        self.rect(x, y, w, h, fill, stroke)
        lines = [lines] if isinstance(lines, str) else lines
        for i, label in enumerate(lines):
            self.text(x+w/2, y+h/2+(i-(len(lines)-1)/2)*25, label,
                      kind='strong teal' if accent else 'strong', anchor='middle')

    def camera(self, x, y, label, sam=False):
        self.rect(x, y, 140, 84, GREEN if sam else PAPER, '#9cbbac' if sam else LINE)
        self.rect(x+32, y+12, 76, 32, '#d7e7dc' if sam else '#e8ece5', 'none', radius=3)
        self.path(f'M {x+38} {y+36} H {x+102}', color='#b9c7bb', width=1)
        self.rect(x+46, y+21, 19, 16, '#558476' if sam else '#9eada1', 'none', radius=2)
        self.parts.append(f'<circle cx="{x+84}" cy="{y+27}" r="9" fill="{TEAL if sam else "#6c8070"}"/>')
        self.text(x+70, y+64, label, kind='strong teal' if sam else 'strong', anchor='middle')

    def save(self, filename):
        svg = '\n'.join(self.parts + ['</svg>']) + '\n'
        (ASSETS / filename).write_text(svg)
        return svg


def extra_view():
    d = Drawing('diagram-extra-view', 652, '03.a · 从输入侧加入 SAM 视角',
        '保留原始主视角与左右腕图像，将主视角的 SAM 渲染图追加为第四张图。四张图共享 ViT 与 visual merger，依次与语言、原有 Depth/Video query 构成 VLM prefix。对 SAM 图比较三种 MRoPE 位置编码与两种 attention mask；动作专家读取 prefix K/V，并根据状态、带噪动作与时间步预测动作。SAM 图在训练和推理时都输入。')
    # Input image slots: the SAM image is derived from the original main view.
    d.path('M 102 116 V 82 H 552 V 116', color=TEAL, arrow=True)
    d.rect(246, 70, 164, 24, fill='white', stroke='none', radius=0)
    d.text(328, 82, 'SAM 前景处理', kind='teal', anchor='middle')
    for x, label, sam in [(32,'主视角原图',False),(182,'左腕原图',False),(332,'右腕原图',False),(482,'SAM 前景图',True)]:
        d.camera(x, 116, label, sam)
        d.path(f'M {x+70} 200 V 230', color=TEAL if sam else MUTED, arrow=True)
    d.box(32, 230, 590, 64, '共享 ViT + Visual merger')
    d.box(632, 230, 140, 64, ['语言', 'Embedding'])
    d.box(782, 230, 146, 64, ['原有 Query', 'Depth / Video'])
    # Prefix: explicit token order, with the SAM view before language.
    d.rect(32, 334, 896, 166, fill='#f6f8f3', stroke='#adbeaf')
    d.path('M 327 294 V 318 H 111 V 350', arrow=True)
    d.path('M 327 318 H 555 V 350', color=TEAL, arrow=True)
    for x in (259,407):
        d.path(f'M {x} 318 V 350', arrow=True)
    for x,end in [(702,703),(855,850)]:
        d.path(f'M {x} 294 V 318 H {end} V 350', arrow=True)
    labels = ['Raw','Wrist L','Wrist R','SAM','Text','Depth / Video']
    for i, label in enumerate(labels):
        d.box(44+i*148, 350, 134 if i<5 else 132, 54, label, accent=i==3)
    d.text(480, 440, 'VLM · Qwen3-VL', kind='strong', anchor='middle')
    d.text(480, 476, 'MRoPE：native / copy / copy_t1    ·    Attention：causal / pair', kind='muted', anchor='middle')
    # Action path: prefix is a conditioning stream, not a visual prediction head.
    d.path('M 640 500 V 546', arrow=True)
    d.text(657, 524, 'Prefix K/V', kind='muted')
    d.box(32, 546, 380, 64, '状态 / 带噪动作 / 时间步 t')
    d.path('M 412 578 H 520', arrow=True)
    d.box(520, 546, 240, 64, 'Action expert')
    d.path('M 760 578 H 808', arrow=True)
    d.box(808, 546, 120, 64, '动作块')
    d.text(32, 632, '新增的 SAM 图在训练与推理时均作为视觉输入。', kind='teal')
    return d.save('03a-input-architecture.svg')


def attention_masks():
    d = Drawing('diagram-attention-mask', 660, '03.a · Attention mask 对照',
        'VLM prefix 的 token 顺序为原始主视角、左腕、右腕、SAM、语言和原有任务 query。矩阵行表示 Query 读取方，列表示 Key 被读取方。causal 为完整的下三角；pair 仅新增原始主视角 Query 读取 SAM Key 的全可见矩形区域。各视角内部仍然为因果注意力。每组用三个代表 token 示意，不表示真实 token 数量；忽略 padding 和动作后缀。')
    d.text(32, 76, '行：Query（读取方）    列：Key（被读取方）', kind='muted')
    groups = ['原图','左腕','右腕','SAM','语言','Query']
    cell, size = 18, 324
    for x, pair in [(108,False),(588,True)]:
        d.text(x-76, 124, 'pair' if pair else 'causal', kind='strong')
        d.text(x+size, 124, '原图 → SAM 整块开放' if pair else 'VLM 原生形式', kind='copper' if pair else 'teal', anchor='end')
        y = 192
        for row in range(18):
            for col in range(18):
                added = pair and row < 3 and 9 <= col < 12
                fill = NEW if added else VISIBLE if col <= row else BLOCKED
                d.rect(x+col*cell, y+row*cell, cell, cell, fill, 'white', radius=0, width=.6)
        for i in range(7):
            p=i*54
            d.path(f'M {x+p} {y} V {y+size}', color='#bac6be', width=.8)
            d.path(f'M {x} {y+p} H {x+size}', color='#bac6be', width=.8)
        for i, label in enumerate(groups):
            d.text(x+i*54+27, 171, label, anchor='middle')
            d.text(x-12, y+i*54+27, label, anchor='end')
        if pair:
            d.rect(x+162, y, 54, 54, fill='none', stroke=COPPER, radius=0, width=2)
        d.text(x-76, 548, '每个原图 token 均可读取全部 SAM token' if pair else 'SAM 可读取原图；原图不可读取 SAM', kind='copper' if pair else 'teal')
    # One shared legend and the minimum abstraction note needed to read the matrix.
    for x, color, label in [(32,VISIBLE,'原有可见区域'),(244,NEW,'pair 新增可见区域'),(510,BLOCKED,'不可见区域')]:
        d.rect(x, 585, 18, 18, color, LINE, radius=2)
        d.text(x+30, 594, label)
    d.text(32, 632, '仅展示 VLM prefix，每组以 3 个 token 示意；视角内部为下三角，跨视角新增区域为整块。', kind='muted')
    return d.save('03a-attention-mask.svg')


def metaquery():
    d = Drawing('diagram-metaquery', 878, '03.b · 用 MetaQuery 学习 SAM 掩码',
        '策略仍输入三张原图。图像和语言 token 后插入可学习 SAM Query，再接原有 Depth/Video query，整个 VLM prefix 使用 causal attention。动作专家可以读取 SAM Query，其在推理中保留。训练专用的 B 型掩码头将主视角 hidden 投影为像素 embedding F，将 SAM Query hidden 投影为 mask embedding e，通过点积及 query 维度最大值形成 32×32 掩码 logits，以离线 SAM 目标计算 BCE 与 Dice loss。推理移除掩码预测头和损失。')
    for x, label in [(32,'主视角原图'),(182,'左腕原图'),(332,'右腕原图')]:
        d.camera(x, 84, label)
        d.path(f'M {x+70} 168 V 192', arrow=True)
    d.box(32, 192, 440, 64, '共享 ViT + Visual merger')
    d.box(482, 192, 140, 64, ['语言', 'Embedding'])
    d.box(632, 192, 140, 64, ['新增可学习', 'SAM Query'], accent=True)
    d.box(782, 192, 146, 64, ['原有 Query', 'Depth / Video'])
    # Prefix queries sit after language. They are not SAM image inputs.
    d.rect(32, 292, 896, 138, fill='#f6f8f3', stroke='#adbeaf')
    d.path('M 252 256 V 278 H 111 V 308', arrow=True)
    d.path('M 252 278 H 407 V 308', arrow=True)
    d.path('M 259 278 V 308', arrow=True)
    for x,end in [(552,555),(702,703),(855,850)]:
        d.path(f'M {x} 256 V 278 H {end} V 308', color=TEAL if x==702 else MUTED, arrow=True)
    for i,label in enumerate(['Raw','Wrist L','Wrist R','Text','SAM Query','Depth / Video']):
        d.box(44+i*148, 308, 134 if i<5 else 132, 54, label, accent=i==4)
    d.text(480, 398, 'VLM · causal attention', kind='strong', anchor='middle')
    # Dashed frame separates the auxiliary training head from the action path.
    d.rect(32, 474, 568, 374, fill='#fffcf8', stroke='#cfae98', dash=True)
    d.path('M 316 430 V 474', color=COPPER, arrow=True)
    d.text(56, 506, '仅训练时 · SAM 辅助监督', kind='strong copper')
    d.box(56, 536, 240, 56, ['主视角 hidden', '像素投影 F'], training=True)
    d.box(320, 536, 256, 56, ['SAM Query hidden', 'Query 投影 e'], training=True)
    d.path('M 176 592 V 614 H 310 V 630', color=COPPER, arrow=True)
    d.path('M 448 592 V 614 H 310', color=COPPER)
    d.parts.append(f'<circle cx="310" cy="648" r="18" fill="{WARM}" stroke="{COPPER}"/>')
    d.text(310, 648, '·', kind='strong copper', anchor='middle')
    d.text(342, 648, '点积 + max over queries', kind='copper')
    d.path('M 310 666 V 682 H 176 V 698', color=COPPER, arrow=True)
    d.box(56, 698, 240, 52, '预测掩码 · 32 × 32', training=True)
    d.box(320, 698, 256, 52, 'SAM 目标 · 32 × 32', training=True)
    d.path('M 176 750 V 772 H 310 V 786', color=COPPER, arrow=True)
    d.path('M 448 750 V 772 H 310', color=COPPER)
    d.box(190, 786, 240, 42, 'BCE + Dice loss', training=True)
    # Prefix attention remains available during action prediction at inference.
    d.path('M 644 430 V 628 H 660', arrow=True)
    d.text(630, 529, 'Prefix K/V', kind='muted', rotate=-90, anchor='middle')
    d.box(660, 474, 268, 64, '状态 / 带噪动作 / t')
    d.path('M 794 538 V 592', arrow=True)
    d.box(660, 592, 268, 72, 'Action expert')
    d.path('M 794 664 V 714', arrow=True)
    d.box(660, 714, 268, 56, '动作块')
    d.rect(660, 794, 268, 54, fill=GREEN, stroke='none')
    d.text(794, 810, '推理保留 SAM Query', kind='teal', anchor='middle')
    d.text(794, 834, '移除掩码预测头与损失', kind='teal', anchor='middle')
    return d.save('03b-metaquery-architecture.svg')


def build_diagrams():
    ASSETS.mkdir(exist_ok=True)
    return {'extra':extra_view(), 'mask':attention_masks(), 'query':metaquery()}


if __name__ == '__main__':
    build_diagrams()
    print('Built three standalone SVG method figures.')
