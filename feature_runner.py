"""Exchange PNGs with the isolated nonfree build; never substitute for SURF."""
import json
import subprocess
from pathlib import Path

import cv2
from core import ROOT


def run_surf_comparison(folder, threshold=400, upright=False, ratio=.75):
    folder = Path(folder)
    python = ROOT / '.surf-venv/Scripts/python.exe'
    if not python.exists():
        raise RuntimeError('SURF environment is missing. Run setup-surf.ps1 before Lab 04.')
    command = [str(python), str(ROOT / 'surf_worker.py'), str(folder),
               str(threshold), str(int(upright)), str(ratio)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if process.returncode:
        raise RuntimeError('Nonfree SURF worker failed: ' + process.stderr[-2000:])
    records = json.loads((folder / 'surf.json').read_text(encoding='utf-8'))
    images = {}
    for algorithm in ('SURF', 'SIFT', 'ORB'):
        for suffix in ('matches', 'keypoints'):
            name = algorithm + '_' + suffix
            images[name] = cv2.imread(str(folder / (name + '.png')))
            if images[name] is None:
                raise RuntimeError('SURF worker did not produce ' + name)
    return images, records
