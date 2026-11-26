import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json
import cv2
import numpy as np
from lab_inputs import EDGE_INPUT,select_input


def test_packaged_jpeg_provenance_and_decoded_input():
    metadata=json.loads(EDGE_INPUT.with_name('input_source.json').read_text())
    assert metadata['instructor_provided'] is False
    assert hashlib.sha256(EDGE_INPUT.read_bytes()).hexdigest()==metadata['sha256']
    actual,source=select_input(2)
    assert np.array_equal(actual,cv2.imread(str(EDGE_INPUT)))
    assert 'replacement' in source


def test_explicit_input_and_sample_override_default(tmp_path):
    supplied=tmp_path/'given.png';im=np.full((12,16,3),30,np.uint8);assert cv2.imwrite(str(supplied),im)
    actual,_=select_input(2,supplied,'Coffee');assert np.array_equal(actual,im)
    actual,source=select_input(2,sample_name='Coffee');assert actual.shape==(400,600,3) and source=='scikit-image:Coffee'
