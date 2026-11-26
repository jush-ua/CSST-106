"""YOLO annotation and source/capture-group splits without duplicate leakage."""
import hashlib
import json
from pathlib import Path
import numpy as np
from core import DATA,CLASSES,decode

def store_photo(raw, session):
    if not session.strip():raise ValueError('Enter a source/capture group name first.')
    image=decode(raw)
    from core import png
    digest=hashlib.sha256(png(image)).hexdigest()
    folder=DATA/'photos';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(digest+'.png')
    if not path.exists():
        path.write_bytes(png(image));(folder/(digest+'.json')).write_text(json.dumps({'session':session.strip(),'boxes':[],'reviewed':False}),encoding='utf-8')
    elif json.loads(path.with_suffix('.json').read_text())['session']!=session.strip():
        raise ValueError('This image already belongs to another session. Do not move duplicates across splits.')
    return path

def save_annotation(path, boxes, reviewed=True):
    p=Path(path);meta=json.loads(p.with_suffix('.json').read_text())
    for b in boxes:
        if len(b)!=5 or not all(np.isfinite(b)) or not float(b[0]).is_integer() or int(b[0]) not in range(3):raise ValueError('Box class must be bottle, cup or phone and all values must be finite.')
        _,x,y,w,h=b
        if min(w,h)<=0 or x-w/2< -1e-6 or y-h/2< -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
            raise ValueError('Box must be inside the image and have positive width and height.')
    meta.update(boxes=boxes,reviewed=reviewed)
    p.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

def build_dataset():
    import shutil
    if (DATA/'dataset'/'frozen.json').exists():raise ValueError('Dataset is frozen for training. Unfreeze it before rebuilding; edited data cannot resume old checkpoints.')
    records=[]
    for p in sorted((DATA/'photos').glob('*.png')):
        meta=json.loads(p.with_suffix('.json').read_text())
        if not meta['reviewed']:raise ValueError(f'Photo {p.stem[:8]} has not been reviewed.')
        save_annotation(p,meta['boxes'])
        records.append((p,meta))
    if not records:raise ValueError('Add and review your photos before building the dataset.')
    sessions=sorted({m['session'] for _,m in records})
    if len(sessions)<3:raise ValueError('At least three source/capture groups are needed for disjoint train/val/test splits.')
    rng=np.random.default_rng(42);rng.shuffle(sessions)
    n=len(sessions);nt=max(1,round(n*.7));nv=max(1,round(n*.15));nt=min(nt,n-2);nv=min(nv,n-nt-1)
    split={s:'train' if i<nt else 'val' if i<nt+nv else 'test' for i,s in enumerate(sessions)}
    target=DATA/'dataset';summary={k:{'images':0,'instances':[0,0,0],'sessions':[],'files':[]} for k in ['train','val','test']}
    # Validate complete split before changing its existing contents.
    for p,m in records:
        item=summary[split[m['session']]];item['images']+=1
        item['files'].append(p.name)
        if m['session'] not in item['sessions']:item['sessions'].append(m['session'])
        for b in m['boxes']:item['instances'][int(b[0])]+=1
    for key,v in summary.items():
        if min(v['instances'])==0:raise ValueError(f'{key} is missing an object class. Add all three classes to more source/capture groups.')
    staging=DATA/'dataset_staging'
    if staging.resolve().parent!=DATA.resolve() or target.resolve().parent!=DATA.resolve():raise ValueError('Dataset staging paths must remain inside the data folder.')
    if staging.exists():shutil.rmtree(staging)
    for p,m in records:
        key=split[m['session']]
        image_dir=staging/'images'/key;label_dir=staging/'labels'/key
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,image_dir/p.name)
        (label_dir/(p.stem+'.txt')).write_text('\n'.join(str(int(b[0]))+' '+' '.join(f'{float(v):.8f}' for v in b[1:]) for b in m['boxes']),encoding='utf-8')
    if target.exists():shutil.rmtree(target)
    staging.rename(target)
    yaml='path: '+target.resolve().as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bottle\n  1: cup\n  2: phone\n'
    (target/'data.yaml').write_text(yaml,encoding='utf-8')
    (target/'split_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (target/'provenance.json').write_text(json.dumps(dataset_provenance(),indent=2),encoding='utf-8')
    return summary

def validate_dataset():
    root=DATA/'dataset';report={};seen=set()
    if not (root/'data.yaml').exists():raise ValueError('No dataset exists. Import and review the collection, then build train/val/test before running the pilot.')
    from ultralytics.utils import YAML
    config=YAML.load(root/'data.yaml')
    if config.get('names')!=dict(enumerate(CLASSES)) or Path(config.get('path','')).resolve()!=root.resolve():raise ValueError('Dataset YAML must reference this dataset and the bottle/cup/phone class mapping.')
    if any(config.get(s)!='images/'+s for s in ('train','val','test')):raise ValueError('Dataset YAML split paths must match the exported session split.')
    manifest=json.loads((root/'split_manifest.json').read_text())
    memberships=[set(manifest[s]['sessions']) for s in ['train','val','test']]
    if any(memberships[i]&memberships[j] for i in range(3) for j in range(i)):raise ValueError('Source/capture groups overlap across splits.')
    for split in ['train','val','test']:
        images=list((root/'images'/split).glob('*.png'));counts=[0,0,0]
        if not images:raise ValueError(f'No images in {split}. Build the dataset first.')
        if set(manifest[split].get('files',[]))!={p.name for p in images}:raise ValueError('Exported images differ from the frozen session manifest. Rebuild the dataset.')
        for p in images:
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen:raise ValueError('Duplicate image found across dataset entries.')
            seen.add(digest);label=root/'labels'/split/(p.stem+'.txt')
            if not label.exists():raise ValueError('Missing label: '+str(label))
            for row in label.read_text().splitlines():
                vals=row.split()
                if len(vals)!=5:raise ValueError('Every label row needs five values.')
                c=int(vals[0]);x,y,w,h=map(float,vals[1:])
                if c not in range(3) or not all(np.isfinite([x,y,w,h])) or min(w,h)<=0 or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
                    raise ValueError('Invalid normalized annotation in '+str(label))
                counts[c]+=1
        if min(counts)==0:raise ValueError(f'{split} must contain all three classes.')
        report[split]={'images':len(images),'instances':dict(zip(CLASSES,counts))}
    return report


def dataset_provenance():
    paths=sorted((DATA/'photos').glob('*.json'))
    sources=[json.loads(p.read_text()) for p in paths]
    public=[r for r in sources if r.get('source_kind')=='coco_public']
    return {'kind':'curated_public' if sources and len(public)==len(sources) else 'personal_or_mixed',
            'classes':CLASSES,'images':len(sources),'public_images':len(public),
            'annotation_source':'COCO instances_val2017; boxes imported, visually reviewed and corrected as needed' if public else 'Locally annotated images',
            'visually_reviewed':sum(bool(r.get('reviewed')) for r in sources),
            'disclosure':'COCO-pretrained weights have prior COCO exposure. This is a classroom fine-tuning exercise, not an independent benchmark.' if public else 'Record each image source accurately.'}


def dataset_fingerprint():
    """Exclude caches and the freeze marker; include every input used for training."""
    root=DATA/'dataset';digest=hashlib.sha256()
    paths=[root/'data.yaml',root/'split_manifest.json']
    if (root/'provenance.json').exists():paths.append(root/'provenance.json')
    paths+=sorted((root/'images').rglob('*.png'))+sorted((root/'labels').rglob('*.txt'))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def freeze_dataset():
    audit=validate_dataset();fingerprint=dataset_fingerprint();path=DATA/'dataset'/'frozen.json'
    if path.exists() and json.loads(path.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Frozen dataset changed. Restore it or unfreeze and start a new experiment.')
    path.write_text(json.dumps({'fingerprint':fingerprint,'audit':audit},indent=2),encoding='utf-8')
    return fingerprint,audit


def capture_progress():
    sessions={}
    for path in sorted((DATA/'photos').glob('*.png')):
        meta=json.loads(path.with_suffix('.json').read_text())
        row=sessions.setdefault(meta['session'],dict(images=0,reviewed=0,backgrounds=0,bottle=0,cup=0,phone=0))
        row['images']+=1;row['reviewed']+=int(meta['reviewed'])
        if meta['reviewed'] and not meta['boxes']:row['backgrounds']+=1
        for cls in {int(b[0]) for b in meta['boxes']}:row[CLASSES[cls]]+=1
    return [{'session':name,**values} for name,values in sorted(sessions.items())]


def contact_sheets():
    """Private labelled thumbnails, paginated to keep all boxes legible."""
    from PIL import Image,ImageDraw
    folder=DATA/'review';folder.mkdir(parents=True,exist_ok=True)
    photos=sorted((DATA/'photos').glob('*.png'));sheets=[]
    for offset in range(0,len(photos),12):
        page=Image.new('RGB',(960,840),'white');draw=ImageDraw.Draw(page)
        for index,path in enumerate(photos[offset:offset+12]):
            meta=json.loads(path.with_suffix('.json').read_text())
            thumb=Image.open(path).convert('RGB');thumb.thumbnail((310,175))
            boxes=ImageDraw.Draw(thumb);w,h=thumb.size
            for cid,x,y,bw,bh in meta['boxes']:
                a=(int((x-bw/2)*w),int((y-bh/2)*h));b=(int((x+bw/2)*w),int((y+bh/2)*h))
                boxes.rectangle((a,b),outline=(0,180,0),width=2)
                label=CLASSES[int(cid)];tx=max(0,min(a[0],w-45));ty=max(0,a[1]-12)
                boxes.rectangle((tx,ty,tx+44,ty+11),fill='black');boxes.text((tx,ty),label,fill='white')
            x=(index%3)*320;y=(index//3)*210
            page.paste(thumb,(x,y))
            draw.text((x,y+177),f"{meta['session'][:24]} | {'reviewed' if meta['reviewed'] else 'UNREVIEWED'}",fill='black')
            draw.text((x,y+192),path.stem[:16],fill='black')
        path=folder/f'contact_{offset//12+1:03d}.png';page.save(path);sheets.append(path)
    return sheets
