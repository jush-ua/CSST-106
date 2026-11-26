"""Reproducible three-class COCO val2017 curation; no full image archive.

python curate_coco.py             # download/import, initially UNREVIEWED
python curate_coco.py --review-ids 139 285  # only after visually inspecting these IDs
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from urllib.request import urlopen
from zipfile import ZipFile

import cv2
import numpy as np
from core import DATA, CLASSES, decode
from dataset import store_photo, save_annotation, contact_sheets

SOURCE='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
ROOT=DATA/'coco_curated'
MAPPING={'bottle':0,'cup':1,'cell phone':2}


def dump(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8',newline='\n')


def download(url,path):
    if path.exists():return path
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            with urlopen(url,timeout=60) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response,stream,length=1024*1024)
            temporary.replace(path);return path
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)


def boxes_for(image,annotations,category_map):
    """COCO JSON IDs are looked up by category name, not YOLO's contiguous IDs."""
    rows=[];w=image['width'];h=image['height']
    for a in annotations:
        if a['category_id'] not in category_map:continue
        if a.get('iscrowd'):return None
        x,y,bw,bh=a['bbox'];x1=max(0.,x);y1=max(0.,y);x2=min(float(w),x+bw);y2=min(float(h),y+bh)
        if x2<=x1 or y2<=y1:return None
        rows.append([category_map[a['category_id']],(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h])
    return rows


def choose_images(images,annotations,categories,positive_target=100,backgrounds=30,seed=42):
    mapping={c['id']:MAPPING[c['name']] for c in categories if c['name'] in MAPPING}
    if set(mapping.values())!={0,1,2}:raise ValueError('Missing expected source categories.')
    by_image=defaultdict(list)
    for a in annotations:by_image[a['image_id']].append(a)
    usable={};pools={c:[] for c in range(3)};negatives=[]
    for image in sorted(images,key=lambda r:r['id']):
        rows=boxes_for(image,by_image[image['id']],mapping)
        if rows is None or len(rows)>8:continue
        usable[image['id']]=(image,rows)
        if not rows:negatives.append(image['id'])
        for cls in {int(r[0]) for r in rows}:
            if any(int(r[0])==cls and r[3]*r[4]>=.005 and min(r[3]*image['width'],r[4]*image['height'])>=8 for r in rows):
                pools[cls].append(image['id'])
    rng=random.Random(seed)
    for pool in pools.values():rng.shuffle(pool)
    rng.shuffle(negatives)
    selected=set();counts=[0,0,0]
    while min(counts)<positive_target:
        cls=min((c for c in range(3) if counts[c]<positive_target),key=lambda c:counts[c])
        while pools[cls] and pools[cls][-1] in selected:pools[cls].pop()
        if not pools[cls]:raise ValueError(f'Not enough eligible images for class {CLASSES[cls]}.')
        identity=pools[cls].pop();selected.add(identity)
        for present in {int(r[0]) for r in usable[identity][1]}:counts[present]+=1
    selected.update(negatives[:backgrounds])
    if len(negatives)<backgrounds:raise ValueError('Not enough background images.')
    # Reserve separate, unseen IDs for qualitative demonstration, not model selection.
    remaining=[i for i,(_,rows) in usable.items() if rows and i not in selected]
    rng.shuffle(remaining);demo=[]
    for cls in (0,1,2,0,1,2,0,1,2,0):
        identity=next(i for i in remaining if i not in demo and any(int(r[0])==cls for r in usable[i][1]))
        demo.append(identity)
    return usable,sorted(selected),demo,counts


def dhash(image):
    small=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(9,8))
    return int(''.join('1' if b else '0' for b in (small[:,1:]>small[:,:-1]).ravel()),2)


def curate():
    if (DATA/'dataset/frozen.json').exists():raise ValueError('A dataset is frozen; preserve it instead of replacing the collection.')
    existing=list((DATA/'photos').glob('*.png'))
    if existing and not (ROOT/'manifest.json').exists():raise ValueError('Existing imported photos need migration; not overwriting them.')
    ROOT.mkdir(parents=True,exist_ok=True)
    archive=download(SOURCE,ROOT/'annotations_trainval2017.zip')
    print('Annotation archive ready.',flush=True)
    with ZipFile(archive) as z:source=json.loads(z.read('annotations/instances_val2017.json'))
    usable,selected,demos,counts=choose_images(source['images'],source['annotations'],source['categories'])
    licences={r['id']:r for r in source['licenses']}
    def fetch(identity):
        image,rows=usable[identity]
        url='http://images.cocodataset.org/val2017/'+image['file_name']
        path=download(url,ROOT/'originals'/image['file_name']);decoded=decode(path.read_bytes())
        if decoded.shape[:2]!=(image['height'],image['width']):raise ValueError('Source dimensions differ from decoded image.')
        return identity,{'image_id':identity,'url':url,'flickr_url':image.get('flickr_url'),
                         'license':licences[image['license']],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'file':str(path.relative_to(ROOT)).replace('\\','/'),'boxes':rows,'dhash':dhash(decoded)}
    with ThreadPoolExecutor(max_workers=8) as pool:records=dict(pool.map(fetch,selected+demos))
    # Group near-duplicates before splitting. Demo must be separate even perceptually.
    parent={i:i for i in selected}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    near=[]
    for n,a in enumerate(selected):
        for b in selected[:n]:
            if (records[a]['dhash']^records[b]['dhash']).bit_count()<=4:
                parent[find(a)]=find(b);near.append([a,b])
    rejected_demo=[i for i in demos if any((records[i]['dhash']^records[j]['dhash']).bit_count()<=4 for j in selected)]
    if rejected_demo:raise ValueError(f'Reserved demo images are near-duplicates; change the deterministic selection: {rejected_demo}')
    for identity in selected:
        record=records[identity];path=ROOT/record['file'];group='coco-val2017-'+str(find(identity))
        photo=store_photo(path.read_bytes(),group)
        meta=json.loads(photo.with_suffix('.json').read_text())
        if meta.get('source',{}).get('image_id') not in (None,identity):raise ValueError('Duplicate pixels under multiple COCO IDs require curation.')
        meta.update(source_kind='coco_public',source={k:v for k,v in record.items() if k not in ('boxes','dhash')},
                    class_names=CLASSES,annotation_source='COCO instances_val2017, remapped and clipped',source_boxes=record['boxes'])
        photo.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        if not meta['reviewed']:save_annotation(photo,record['boxes'],reviewed=False)
        record['imported_file']=photo.name;record['source_group']=group
    manifest={'dataset':'COCO 2017 validation curated collection','annotation_url':SOURCE,
              'annotation_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'seed':42,'target_positive_images_per_class':100,'background_images':30,
              'actual_positive_image_counts':dict(zip(CLASSES,counts)),'classes':CLASSES,
              'selected_ids':selected,'demo_ids':demos,'near_duplicate_pairs':near,
              'images':[records[i] for i in selected+demos],
              'disclosure':'Public images; existing COCO boxes imported for visual review/correction. Not personal photography. COCO-pretrained weights have prior COCO exposure; this is a classroom fine-tuning exercise, not an independent benchmark.'}
    dump(ROOT/'manifest.json',manifest);contact_sheets()
    print(json.dumps({'images':len(selected),'demo_images':len(demos),'positives':dict(zip(CLASSES,counts)),'near_duplicate_pairs':near},indent=2),flush=True)


def mark_reviewed(identities):
    """Called explicitly only after the requested image IDs have been visually inspected."""
    manifest=json.loads((ROOT/'manifest.json').read_text())
    rows={r['image_id']:r for r in manifest['images'] if r['image_id'] in manifest['selected_ids']}
    for identity in identities:
        row=rows[identity];path=DATA/'photos'/row['imported_file'];meta=json.loads(path.with_suffix('.json').read_text())
        save_annotation(path,meta['boxes'],True)
        meta=json.loads(path.with_suffix('.json').read_text());meta['review']={'method':'Visual inspection of labelled contact sheet; source image inspection where needed','reviewer':'AI-assisted visual review','time':time.time(),'source_boxes_changed':meta['boxes']!=meta['source_boxes']}
        dump(path.with_suffix('.json'),meta)
    print('Reviewed IDs:',identities)


def publish_provenance(destination):
    """Publish source records and split IDs, never image pixels or per-image labels."""
    manifest=json.loads((ROOT/'manifest.json').read_text())
    split=json.loads((DATA/'dataset/split_manifest.json').read_text())
    membership={name:key for key,row in split.items() for name in row['files']}
    records=[]
    for row in manifest['images']:
        entry={k:row[k] for k in ('image_id','url','flickr_url','license','sha256')}
        if row['image_id'] in manifest['selected_ids']:
            meta=json.loads((DATA/'photos'/row['imported_file']).with_suffix('.json').read_text())
            review=meta.get('review')
            if review and review.get('reviewer')=='assistant':review=dict(review,reviewer='AI-assisted visual review')
            entry.update(split=membership[row['imported_file']],source_group=meta['session'],reviewed=meta['reviewed'],review=review,curation_note=meta.get('curation_note'))
        else:entry['split']='demo'
        records.append(entry)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    dump(destination/'curation.json',{'dataset':manifest['dataset'],'classes':manifest['classes'],
         'annotation_url':manifest['annotation_url'],'annotation_archive_sha256':manifest['annotation_archive_sha256'],
         'seed':manifest['seed'],'disclosure':manifest['disclosure'],'images':records})
    return destination/'curation.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review-ids',nargs='+',type=int)
    args=parser.parse_args()
    if args.review_ids:mark_reviewed(args.review_ids)
    else:curate()


if __name__=='__main__':main()
