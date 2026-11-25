import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
from core import manipulate,edges,stretch,contrast,features,region_grow,mask_scores,segmentation,decode

def test_geometry_and_bgr_edit_do_not_mutate_input():
    a=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3);before=a.copy()
    out=manipulate(a,row=1,col=2,bgr=(3,5,7),roi=(1,1,3,2),size=(2,3))
    assert np.array_equal(a,before)
    assert tuple(out['edited'][1,2])==(3,5,7)
    assert np.array_equal(out['crop'],a[1:3,1:4])
    assert out['resized'].shape==(3,2,3)
    assert np.array_equal(out['rotated'][0,-1],a[0,0])

def test_invalid_crop():
    with pytest.raises(ValueError):manipulate(np.zeros((4,4,3),np.uint8),row=0,col=0,bgr=(0,0,0),roi=(3,3,2,2),size=(2,2))

def test_edge_baseline_matches_source_operations():
    a=np.random.default_rng(42).integers(0,256,(30,40),dtype=np.uint8)
    out,raw=edges(a);blur=cv2.GaussianBlur(a,(3,3),0)
    assert np.array_equal(out['canny'],cv2.Canny(blur,100,200))
    assert np.allclose(raw['magnitude'],np.hypot(raw['gx'],raw['gy']))
    assert raw['gx'].dtype==np.float64

def test_invalid_canny_thresholds():
    with pytest.raises(ValueError):edges(np.zeros((4,4),np.uint8),low=200,high=100)

def test_constant_contrast_is_finite_and_unchanged():
    a=np.full((10,10),80,np.uint8);assert np.array_equal(stretch(a),a)
    b=stretch(np.array([[100,150,200]],np.uint8));assert b.min()==0 and b.max()==255

def test_colour_equalization_preserves_shape():
    a=np.random.default_rng(42).integers(0,256,(24,30,3),dtype=np.uint8)
    out,channels=contrast(a,colour=True)
    assert all(x.shape==a.shape for x in out.values())
    assert all(x.shape==a.shape[:2] for x in channels.values())

def test_no_features_does_not_crash():
    a=np.zeros((64,64,3),np.uint8)
    for algorithm in ['SIFT','ORB']:
        output,rich,stats=features(a,a,algorithm)
        assert stats['accepted_matches']==0

def test_seed_value_and_connectivity():
    a=np.array([[10,100,100],[100,10,100],[100,100,10]],np.uint8)
    assert region_grow(a,(0,0),0,4).sum()==1
    assert region_grow(a,(0,0),0,8).sum()==3

def test_region_tolerance_inclusive():
    a=np.array([[10,15,20]],np.uint8)
    assert region_grow(a,(0,0),5).tolist()==[[True,True,False]]

def test_mask_scores_ignore_void_and_empty_classes():
    a=np.array([[0,1],[1,1]]);t=np.array([[0,1],[0,255]])
    r=mask_scores(a,t,3)
    assert r['pixel_accuracy']==pytest.approx(2/3)
    assert r['classes'][1]['iou']==.5
    assert r['classes'][2]['iou'] is None

def test_mask_shapes_are_not_silently_resized():
    with pytest.raises(ValueError):mask_scores(np.zeros((2,2)),np.zeros((3,3)),2)

def test_empty_watershed():
    out,markers,stats=segmentation(np.zeros((20,20,3),np.uint8),(0,0))
    assert stats['watershed_seed_components']==0
    assert markers.shape==(20,20)

def test_unreadable_upload():
    with pytest.raises(ValueError):decode(b'not an image')
