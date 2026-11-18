#!/usr/bin/env python3
"""Extract paired video frames and render a self-contained SVG comparison plate.

Run: python3 WebPage/source/draw_sam_examples.py
Frames retain their original field of view and colors; no mask is reconstructed.
"""
from pathlib import Path
import base64
import hashlib
import json

from draw_diagrams import Drawing, ASSETS, LINE

ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = 'robotwin_v30_modlr_no_wristaug_datafix_30000_demo_randomized_samvla_judge_s0.1_w2_ep10_20261026_062044'
RUN = ROOT / 'VLABenchmarkResult' / RUN_NAME
SELECTED = [
    ('adjust_bottle', 'adjust_bottle'),
    ('blocks_ranking_size', 'blocks_ranking_size'),
    ('lift_pot', 'lift_pot'),
    ('move_pillbottle_pad', 'move_pillbottle_pad'),
    ('place_object_basket', 'place_object_basket'),
    ('place_shoe', 'place_shoe'),
]
EPISODE = 0
FRAME = 0


def extract_examples():
    import cv2
    cv2.setNumThreads(1)
    dest = ASSETS / 'sam-examples'
    dest.mkdir(exist_ok=True)
    rows = []
    for task, label in SELECTED:
        task_dir = RUN / 'eval_results' / task
        matches = sorted(task_dir.glob(f'episode{EPISODE}_*_cam_high_masked.mp4'))
        if len(matches) != 1:
            raise ValueError(f'Expected one paired episode for {task}: {matches}')
        masked = matches[0]
        raw = masked.with_name(masked.name.replace('_cam_high_masked', ''))
        meta_path = masked.with_suffix('.json')
        meta = json.loads(meta_path.read_text())
        assert meta['episode'] == EPISODE
        assert meta['outside_scale'] == 0.1 and meta['edge_width'] == 2 and meta['judge']
        assert not meta['fallback']
        frames = {}
        for kind, path in [('raw',raw),('sam',masked)]:
            cap = cv2.VideoCapture(str(path))
            try:
                count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise ValueError(f'Cannot read first frame: {path}')
            h, w = frame.shape[:2]
            png = dest / f'{task}-{kind}.png'
            assert cv2.imwrite(str(png), frame)
            frames[kind] = {
                'video': str(path.relative_to(ROOT)), 'frame_index': FRAME,
                'video_frames': count, 'fps': fps, 'width': w, 'height': h,
                'png': str(png.relative_to(ROOT)),
                'sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
            }
        assert (frames['raw']['width'],frames['raw']['height']) == (frames['sam']['width'],frames['sam']['height'])
        rows.append({'task':task, 'label':label, 'episode':EPISODE,
                     'instruction':meta['instruction'], 'metadata':str(meta_path.relative_to(ROOT)),
                     'mask_area_fraction':meta['mask_area_fraction'][FRAME], 'frames':frames})
    manifest = {
        'run':RUN_NAME, 'scene':'randomized', 'outside_scale':0.1, 'edge_width':2, 'judge':True,
        'frame_pairing':'Both videos start from the same initial cam_high observation: SAM begin() precedes the first take_action(), whose recorder writes cached now_obs before applying the action.',
        'processing':'Decode frame 0 from each original video; save lossless PNG; no crop, color adjustment, or mask regeneration. SVG displays the full frame at the same aspect ratio.',
        'examples':rows,
    }
    (ASSETS.parent / 'data' / 'sam-mask-examples.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return rows


def build_sam_examples():
    rows = extract_examples()
    d = Drawing('diagram-sam-examples', 832, '随机场景中的 SAM 前景输入',
        '六个任务的原图与 SAM 前景图对照：adjust_bottle、blocks_ranking_size、lift_pot、move_pillbottle_pad、place_object_basket和place_shoe。每组左侧为原图，右侧为同一场景同一时刻的 SAM 输入；保留任务相关物体及目标区域，将背景亮度降到 0.1 并添加 2px 描边。')
    d.text(928, 40, '背景 ×0.1 · 边缘 2px', kind='teal', anchor='end')
    for i, row in enumerate(rows):
        x = 32 + (i % 2)*464
        y = 88 + (i // 2)*248
        d.text(x, y, chr(65+i), kind='strong copper')
        d.text(x+28, y, row['label'], kind='strong')
        # Same type size for every panel label. Use the canonical English benchmark task names.
        d.text(x+105, y+32, '原始图像', kind='muted', anchor='middle')
        d.text(x+327, y+32, 'SAM 前景图', kind='teal', anchor='middle')
        for j, kind in enumerate(['raw','sam']):
            frame = row['frames'][kind]
            png = ROOT / frame['png']
            xx, yy, w = x+j*222, y+52, 210
            h = w*frame['height']/frame['width']
            data = base64.b64encode(png.read_bytes()).decode('ascii')
            d.parts.append(f'<image x="{xx}" y="{yy}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{data}"/>')
            d.rect(xx, yy, w, h, fill='none', stroke=LINE, radius=0, width=.8)
    return d.save('03-sam-mask-examples.svg')


if __name__ == '__main__':
    build_sam_examples()
    print('Built SAM comparison figure from six synchronized raw/SAM video pairs.')
