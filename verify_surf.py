"""Generate measured Lab 04 evidence using the isolated nonfree build."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
from skimage import data

ROOT = Path(__file__).resolve().parent


def main():
    python = ROOT / '.surf-venv/Scripts/python.exe'
    image = cv2.cvtColor(data.camera(), cv2.COLOR_GRAY2BGR)
    output = ROOT / 'examples/lab04'
    analysis = [
        '# Lab 04 measured SURF comparison', '',
        'Public scikit-image Camera sample, 512 x 512. Matching uses two nearest neighbours and a 0.75 ratio test. '
        'SURF/SIFT use L2 distance; ORB uses Hamming distance. All rows use the same nonfree OpenCV 4.12.0 build.', '',
        'Descriptors represent local appearance around detected keypoints so corresponding regions can be compared. '
        'The default SURF descriptor has 64 floating-point dimensions; SIFT has 128 and ORB has 32 bytes of binary tests. '
        'Accepted match counts measure ratio-test acceptance, not verified geometric correctness.', '',
    ]
    for case in ('rotation', 'upright_translation'):
        folder = output / case
        folder.mkdir(parents=True, exist_ok=True)
        if case == 'rotation':
            other = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            import numpy as np
            other = cv2.warpAffine(image, np.float32([[1, 0, 20], [0, 1, 15]]), (512, 512))
        cv2.imwrite(str(folder / 'a.png'), image)
        cv2.imwrite(str(folder / 'b.png'), other)
        subprocess.run([str(python), str(ROOT / 'surf_worker.py'), str(folder), '400', '0', '.75'], check=True)
        records = json.loads((folder / 'surf.json').read_text())
        manifest = {
            'source': 'scikit-image Camera public sample', 'case': case,
            'transform': '90 degrees clockwise' if case == 'rotation' else 'translation x=20,y=15 pixels, black border',
            'ratio': .75, 'comparisons': records,
            'timing_note': 'Single CPU run; timings include both images and are not a general benchmark.',
            'artifacts': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.png'))},
        }
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        analysis += [
            '## ' + case.replace('_', ' ').title(), '',
            manifest['transform'] + '.', '',
            '| Detector | Hessian | Upright | Keypoints A/B | Matches | Detect + describe ms | Match ms |',
            '|---|---:|---|---:|---:|---:|---:|',
        ]
        for row in records:
            analysis.append(
                f"| {row['algorithm']} | {row.get('hessian_threshold', 'n/a')} | {row.get('upright', 'n/a')} | "
                f"{row['keypoints_a']}/{row['keypoints_b']} | {row['accepted_matches']} | "
                f"{row['detect_describe_ms']:.2f} | {row['match_ms']:.2f} |"
            )
        surf = [r for r in records if r['algorithm'] == 'SURF' and not r['upright']]
        analysis += ['', 'Measured threshold sweep (A/B keypoints): ' + '; '.join(
            f"{r['hessian_threshold']}: {r['keypoints_a']}/{r['keypoints_b']}" for r in surf
        ) + '.', '']
        print(case, json.dumps(records, indent=2))
    analysis += [
        '## Interpretation', '',
        'Raising the Hessian threshold rejects weaker detector responses and generally leaves fewer keypoints. '
        'Inspect the measured counts above for this image. Fewer keypoints can reduce matching work while losing useful correspondences.', '',
        'U-SURF skips dominant-orientation estimation. It is useful when camera/object orientation stays fixed, '
        'as in the translation case, but gives up descriptor rotation invariance. Compare its accepted matches '
        'at threshold 400 in both cases rather than treating every extra match as a correct correspondence.', '',
        'SURF approximates Hessian responses with box filters/integral images and describes Haar-wavelet responses. '
        'SIFT uses scale-space extrema and gradient histograms; ORB uses oriented FAST and binary descriptors. '
        'The tables record this machine\'s measured detection/description and matching costs for each method. '
        'They are single CPU runs, include both images, may contain warm-up effects and do not establish a universal speed ranking. '
        'SIFT/ORB are capped at 1,500 keypoints by this implementation; SURF uses its Hessian threshold instead.', '',
    ]
    (output / 'ANALYSIS.md').write_text('\n'.join(analysis), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps({
        'lab': 'lab04', 'state': 'completed',
        'cases': ['rotation/manifest.json', 'upright_translation/manifest.json'],
        'build': 'OpenCV 4.12.0, OPENCV_ENABLE_NONFREE=ON; all algorithms in the same isolated build',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
