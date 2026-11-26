"""Exercise the real CLI-to-nonfree-worker boundary when that build exists."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / '.surf-venv/Scripts/python.exe').exists(), reason='Optional nonfree source build is not installed')
def test_standalone_lab04_exports_actual_surf_and_requested_parameters(tmp_path):
    image = np.random.default_rng(42).integers(0, 256, (192, 192, 3), dtype=np.uint8)
    source = tmp_path / 'features.png'
    assert cv2.imwrite(str(source), image)
    process = subprocess.run([sys.executable, str(ROOT / 'run_lab.py'), '4', '--image', str(source),
                              '--hessian', '321', '--upright', '--ratio', '.7'],
                             cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stderr
    folder = Path(process.stdout.strip().splitlines()[-1])
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['parameters']['threshold'] == 321
    assert manifest['parameters']['upright'] is True
    assert manifest['parameters']['ratio'] == .7
    rows = manifest['metrics']['comparisons']
    assert {r['algorithm'] for r in rows} == {'SURF', 'SIFT', 'ORB'}
    assert len({r['opencv_version'] for r in rows}) == 1
    surf = [r for r in rows if r['algorithm'] == 'SURF']
    assert {(r['hessian_threshold'], r['upright']) for r in surf} == {
        (threshold, upright) for threshold in (100, 321, 400, 1000) for upright in (False, True)
    }
    assert all(r['descriptor_dimensions'] == 64 for r in surf)
    assert cv2.imread(str(folder / 'SURF_matches.png')) is not None
