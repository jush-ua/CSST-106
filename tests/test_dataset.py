import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pytest
import dataset
from core import png

def test_session_split_and_label_integrity(tmp_path,monkeypatch):
    monkeypatch.setattr(dataset,'DATA',tmp_path)
    for i in range(10):
        image=np.full((10,10,3),i,np.uint8)
        p=dataset.store_photo(png(image),'session-'+str(i))
        dataset.save_annotation(p,[[c,.5,.5,.3,.3] for c in range(3)])
    summary=dataset.build_dataset();report=dataset.validate_dataset()
    sets=[set(v['sessions']) for v in summary.values()]
    assert not sets[0]&sets[1] and not sets[1]&sets[2] and not sets[0]&sets[2]
    assert sum(v['images'] for v in report.values())==10

def test_invalid_box_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(dataset,'DATA',tmp_path)
    p=dataset.store_photo(png(np.zeros((10,10,3),np.uint8)),'a')
    with pytest.raises(ValueError):dataset.save_annotation(p,[[0,.9,.9,.5,.5]])

def test_missing_review_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(dataset,'DATA',tmp_path)
    dataset.store_photo(png(np.zeros((10,10,3),np.uint8)),'a')
    with pytest.raises(ValueError,match='not been reviewed'):dataset.build_dataset()
