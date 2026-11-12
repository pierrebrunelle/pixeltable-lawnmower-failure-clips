"""Drive the mower failure-clip pipeline: ingest runs, label clips, mine look-alikes.

    python make_videos.py
    pxt schema update app.py mower
    python run.py                 # ingest the fleet's runs + print the review report
    python run.py --report        # report only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pixeltable as pxt

HERE = Path(__file__).parent
TARGET = os.environ.get('MOWER_TARGET', 'mower')

FLEET = [
    ('MW-101', 'Atlas-2', 'Elm St backyard', '4.2.1'),
    ('MW-102', 'Atlas-2', 'Hillside orchard', '4.2.1'),
    ('MW-103', 'Atlas-3', 'Cedar Ave front lawn', '5.0.0'),
    ('MW-104', 'Atlas-3', 'Hillside orchard', '5.0.0'),
    ('MW-105', 'Atlas-2', 'Park strip', '4.1.9'),
    ('MW-106', 'Atlas-3', 'Park strip', '5.0.0'),  # camera lost power: truncated upload
    ('MW-107', 'Atlas-3', 'Cedar Ave front lawn', None),  # run aborted before recording
]


def video(run_id: str) -> str | None:
    p = HERE / 'data' / 'runs' / f'{run_id}.mp4'
    return str(p) if p.exists() else None


def ingest() -> None:
    runs = pxt.get_table(f'{TARGET}/runs')
    cols = set(runs.columns())
    rows = []
    for run_id, mower, yard, firmware in FLEET:
        row = {'run_id': run_id, 'mower': mower, 'yard': yard, 'video': video(run_id)}
        if 'firmware' in cols:
            row['firmware'] = firmware
        rows.append(row)
    status = runs.insert(rows, on_error='ignore')
    print(f'runs: {status.num_rows} rows written, {status.num_excs} cell errors')


def report() -> None:
    runs = pxt.get_table(f'{TARGET}/runs')
    clips = pxt.get_table(f'{TARGET}/clips')
    print(runs.select(runs.run_id, runs.yard, runs.run_secs).order_by(runs.run_id).collect())
    errs = runs.where(runs.video.errortype != None).select(runs.run_id, runs.video.errortype).collect()  # noqa: E711
    print(f'unreadable uploads: {[r["run_id"] for r in errs]}')
    cols = set(clips.columns())
    sel = [clips.run_id, clips.segment_start, clips.segment_end]
    if 'label' in cols:
        sel += [clips.label]
    if 'brightness' in cols:
        sel += [clips.brightness]
    print(clips.select(*sel).order_by(clips.run_id, clips.segment_start).collect())
    if 'label' in cols:
        print(clips.group_by(clips.label).select(clips.label, n=pxt.functions.count(clips.pos)).order_by(clips.label).collect())


def mine() -> None:
    """Mine look-alike failure clips: rule queue, CLIP text search, CLIP search by example frame."""
    clips = pxt.get_table(f'{TARGET}/clips')
    for label in ('hose_snag', 'tip_slope', 'stall_dark', 'fence_bump'):
        q = clips.where(clips.label == label).order_by(clips.run_id, clips.segment_start)
        print(f'queue {label}:', [(r['run_id'], r['segment_start']) for r in q.select(clips.run_id, clips.segment_start).collect()])

    def top(sim, k: int = 4) -> list:
        rows = clips.order_by(sim, asc=False).select(clips.run_id, clips.segment_start, clips.label, score=sim).limit(k).collect()
        return [(r['run_id'], r['segment_start'], r['label'], round(r['score'], 3)) for r in rows]

    for text in ('an orange garden hose lying across green grass', 'blue sky, the camera is tilted', 'a pitch black frame'):
        print(f'text {text!r}:', top(clips.keyframe.similarity(string=text), 3))
    # example frame: the hose snag a field tech flagged on MW-101
    ex = clips.where((clips.run_id == 'MW-101') & (clips.segment_start == 6.0)).select(clips.keyframe).collect()[0]['keyframe']
    print('like MW-101 @ 6 s:', top(clips.keyframe.similarity(image=ex), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--mine', action='store_true', help='also run the look-alike searches')
    a = ap.parse_args()
    if not a.report:
        ingest()
    report()
    if a.mine or not a.report:
        mine()
