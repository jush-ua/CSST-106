"""YOLO annotation and capture-session splits without near-duplicate leakage."""
import hashlib
import json
from pathlib import Path
import numpy as np
from core import DATA,CLASSES,decode

def store_photo(raw, session):
    if not session.strip():raise ValueError('Enter a capture session name first.')
    image=decode(raw);digest=hashlib.sha256(raw).hexdigest()
    folder=DATA/'photos';folder.mkdir(parents=True,exist_ok=True)
    from core import png
    path=folder/(digest+'.png')
    if not path.exists():
        path.write_bytes(png(image));(folder/(digest+'.json')).write_text(json.dumps({'session':session.strip(),'boxes':[],'reviewed':False}),encoding='utf-8')
    return path

def save_annotation(path, boxes, reviewed=True):
    p=Path(path);meta=json.loads(p.with_suffix('.json').read_text())
    for b in boxes:
        if len(b)!=5 or int(b[0]) not in range(3):raise ValueError('Box class must be bottle, mug or phone.')
        _,x,y,w,h=b
        if min(w,h)<=0 or x-w/2< -1e-6 or y-h/2< -1e-6 or x+w/2>1+1e-6 or y+h/2>1+1e-6:
            raise ValueError('Box must be inside the image and have positive width and height.')
    meta.update(boxes=boxes,reviewed=reviewed)
    p.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

def build_dataset():
    import shutil
    records=[]
    for p in sorted((DATA/'photos').glob('*.png')):
        meta=json.loads(p.with_suffix('.json').read_text())
        if not meta['reviewed']:raise ValueError(f'Photo {p.stem[:8]} has not been reviewed.')
        save_annotation(p,meta['boxes'])
        records.append((p,meta))
    if not records:raise ValueError('Add and review your photos before building the dataset.')
    sessions=sorted({m['session'] for _,m in records})
    if len(sessions)<3:raise ValueError('At least three separate capture sessions are needed for disjoint train/val/test splits.')
    rng=np.random.default_rng(42);rng.shuffle(sessions)
    n=len(sessions);nt=max(1,round(n*.7));nv=max(1,round(n*.15));nt=min(nt,n-2);nv=min(nv,n-nt-1)
    split={s:'train' if i<nt else 'val' if i<nt+nv else 'test' for i,s in enumerate(sessions)}
    target=DATA/'dataset';summary={k:{'images':0,'instances':[0,0,0],'sessions':[]} for k in ['train','val','test']}
    # Validate complete split before changing its existing contents.
    for p,m in records:
        item=summary[split[m['session']]];item['images']+=1
        if m['session'] not in item['sessions']:item['sessions'].append(m['session'])
        for b in m['boxes']:item['instances'][int(b[0])]+=1
    for key,v in summary.items():
        if min(v['instances'])==0:raise ValueError(f'{key} is missing an object class. Add all three classes to more capture sessions.')
    staging=DATA/'dataset_staging'
    if staging.exists():shutil.rmtree(staging)
    for p,m in records:
        key=split[m['session']]
        image_dir=staging/'images'/key;label_dir=staging/'labels'/key
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,image_dir/p.name)
        (label_dir/(p.stem+'.txt')).write_text('\n'.join(str(int(b[0]))+' '+' '.join(f'{float(v):.8f}' for v in b[1:]) for b in m['boxes']),encoding='utf-8')
    if target.exists():shutil.rmtree(target)
    staging.rename(target)
    yaml='path: '+target.resolve().as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bottle\n  1: mug\n  2: phone\n'
    (target/'data.yaml').write_text(yaml,encoding='utf-8')
    (target/'split_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

def validate_dataset():
    root=DATA/'dataset';report={};seen=set()
    for split in ['train','val','test']:
        images=list((root/'images'/split).glob('*.png'));counts=[0,0,0]
        if not images:raise ValueError(f'No images in {split}. Build the dataset first.')
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
