"""Explicit input selection and provenance for the student-selected Lab 02 image."""
import hashlib
import json
from importlib.metadata import version
from core import ROOT, decode, sample

EDGE_INPUT = ROOT / 'examples/lab02/input.jpg'


def package_edge_input():
    import cv2
    EDGE_INPUT.parent.mkdir(parents=True, exist_ok=True)
    if not EDGE_INPUT.exists():
        if not cv2.imwrite(str(EDGE_INPUT), sample('Camera'), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError('Could not write Lab 02 input.jpg.')
    provenance = {
        'file': 'input.jpg', 'instructor_provided': False,
        'source': 'Public scikit-image data.camera sample',
        'scikit_image_version': version('scikit-image'), 'jpeg_quality': 95,
        'disclosure': 'Student-selected input.jpg; students choose their image for this activity.',
        'sha256': hashlib.sha256(EDGE_INPUT.read_bytes()).hexdigest(),
    }
    (EDGE_INPUT.parent / 'input_source.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8', newline='\n')
    return EDGE_INPUT


def select_input(lab, image_path=None, sample_name=None):
    """Explicit file wins, then explicit sample, then the lab default."""
    if image_path:
        from pathlib import Path
        path = Path(image_path)
        return decode(path.read_bytes()), str(path)
    if sample_name:
        return sample(sample_name), 'scikit-image:' + sample_name
    if lab == 2:
        return decode(EDGE_INPUT.read_bytes()), 'Packaged input.jpg (student-selected scikit-image Camera image)'
    return sample('Astronaut'), 'scikit-image:Astronaut'
