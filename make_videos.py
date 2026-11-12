"""Generate small synthetic mower-camera videos with ffmpeg (offline, ~1 MB total).

Each run is a sequence of 3-second scenes from the mower's front camera:
normal mowing (textured green grass) plus the failure moments the fleet team
wants to mine (garden hose caught in the deck, tipping on a slope, stalling in
the dark under a deck, bumping a fence). One file is deliberately truncated so
the pipeline's error handling has something real to deal with.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / 'data' / 'runs'
W, H, FPS, SCENE = 320, 240, 10, 3

SCENES = {
    # textured grass: green base + noise
    'grass': f'color=c=0x2e8b3a:s={W}x{H}:r={FPS}:d={SCENE},noise=alls=40',
    # orange garden hose crossing the frame
    'hose': f'color=c=0x2e8b3a:s={W}x{H}:r={FPS}:d={SCENE},noise=alls=30,'
    f'drawbox=x=0:y=100:w={W}:h=28:color=0xff7f00:t=fill',
    # tipping on a slope: sky fills the top half, horizon tilted
    'tip': f'color=c=0x87ceeb:s={W}x{H}:r={FPS}:d={SCENE},'
    f'drawbox=x=0:y=150:w={W}:h=90:color=0x2e8b3a:t=fill,rotate=0.5:fillcolor=0x87ceeb',
    # stalled in the dark under a deck
    'dark': f'color=c=0x0a0a0a:s={W}x{H}:r={FPS}:d={SCENE},noise=alls=10',
    # fence: brown vertical slats
    'fence': f'color=c=0x2e8b3a:s={W}x{H}:r={FPS}:d={SCENE},'
    + ','.join(f'drawbox=x={x}:y=20:w=24:h=200:color=0x8b5a2b:t=fill' for x in range(10, W, 50)),
}

RUNS = {
    'MW-101': ['grass', 'grass', 'hose', 'grass'],
    'MW-102': ['grass', 'tip', 'grass', 'grass'],
    'MW-103': ['grass', 'hose', 'grass', 'dark'],
    'MW-104': ['fence', 'grass', 'grass', 'tip'],
    'MW-105': ['grass', 'grass', 'grass', 'grass'],
}


def render(run_id: str, scenes: list[str]) -> Path:
    out = OUT / f'{run_id}.mp4'
    inputs: list[str] = []
    for s in scenes:
        inputs += ['-f', 'lavfi', '-i', SCENES[s]]
    n = len(scenes)
    concat = ''.join(f'[{i}:v]' for i in range(n)) + f'concat=n={n}:v=1:a=0,format=yuv420p[v]'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', *inputs, '-filter_complex', concat, '-map', '[v]',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '32', '-g', str(FPS), str(out)]
    subprocess.run(cmd, check=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for run_id, scenes in RUNS.items():
        p = render(run_id, scenes)
        print(f'{p.name}: {" -> ".join(scenes)} ({p.stat().st_size // 1024} KB)')
    # a truncated upload (camera lost power mid-write)
    good = (OUT / 'MW-105.mp4').read_bytes()
    (OUT / 'MW-106.mp4').write_bytes(good[: len(good) // 5])
    print('MW-106.mp4: truncated copy of MW-105 (corrupt on purpose)')


if __name__ == '__main__':
    main()
