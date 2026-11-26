import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from curate_coco import boxes_for,choose_images,dhash


def test_source_ids_are_remapped_by_category_names_and_boxes_clipped():
    image={'width':100,'height':80}
    rows=boxes_for(image,[{'category_id':44,'bbox':[-10,20,50,80],'iscrowd':0},
                         {'category_id':999,'bbox':[0,0,10,10]}],{44:0})
    assert rows==[[0,.2,.625,.4,.75]]
    assert boxes_for(image,[{'category_id':44,'bbox':[0,0,10,10],'iscrowd':1}],{44:0}) is None


def test_selection_backgrounds_demo_and_reproducibility():
    images=[{'id':i,'width':100,'height':100} for i in range(35)]
    categories=[{'id':101,'name':'bottle'},{'id':202,'name':'cup'},{'id':303,'name':'cell phone'}]
    annotations=[{'image_id':i,'category_id':[101,202,303][i%3],'bbox':[10,10,40,40],'iscrowd':int(i==0)} for i in range(30)]
    usable,selected,demo,counts=choose_images(images,annotations,categories,positive_target=2,backgrounds=1)
    assert 0 not in usable
    assert min(counts)>=2 and len(demo)==10
    assert not set(selected)&set(demo) and len(set(selected))==len(selected)
    negatives=[i for i in selected if not usable[i][1]]
    assert len(negatives)==1 and negatives[0]>=30
    assert (selected,demo)==choose_images(images,annotations,categories,2,1)[1:3]


def test_perceptual_hash_handles_identical_images():
    image=np.random.default_rng(42).integers(0,255,(50,80,3),dtype=np.uint8)
    assert dhash(image)==dhash(image.copy())
