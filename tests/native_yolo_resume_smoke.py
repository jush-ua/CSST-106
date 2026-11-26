"""Optional real-library checkpoint smoke test, never a personal Lab 06 result.

Run manually: .venv\Scripts\python.exe tests\native_yolo_resume_smoke.py
Synthetic fixtures live only in ignored runs/verification_yolo_resume_*.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import json
import shutil
import time
import uuid
import cv2
import numpy as np
from core import DATA,RUNS
from lab06 import TrainingPaused


def main():
    import torch
    from ultralytics import YOLO
    folder=RUNS/('verification_yolo_resume_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True)
    dataset=folder/'synthetic'
    for split,count in [('train',6),('val',3)]:
        images=dataset/'images'/split;labels=dataset/'labels'/split
        images.mkdir(parents=True);labels.mkdir(parents=True)
        for i in range(count):
            image=np.full((96,96,3),30+i,np.uint8)
            rows=[]
            for cls,(x,y) in enumerate([(20,24),(65,24),(45,65)]):
                cv2.rectangle(image,(x-8,y-10),(x+8,y+10),(50+cls*80,200-cls*50,100+i*10),-1)
                rows.append(f'{cls} {x/96} {y/96} {16/96} {20/96}')
            cv2.imwrite(str(images/f'{i}.png'),image)
            (labels/f'{i}.txt').write_text('\n'.join(rows))
    yaml=dataset/'data.yaml'
    yaml.write_text(f'path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: bottle\n  1: mug\n  2: phone\n')
    checkpoint=folder/'resume.pt'
    model=YOLO(str(DATA/'models/yolo11n.pt'))
    def pause(trainer):
        shutil.copyfile(trainer.last,checkpoint)
        raise TrainingPaused()
    model.add_callback('on_model_save',pause)
    try:
        model.train(data=str(yaml),epochs=3,imgsz=64,batch=2,device='cpu',workers=0,
                    patience=0,seed=42,plots=False,save=True,save_period=-1,
                    project=str(folder),name='native',exist_ok=True)
    except TrainingPaused:pass
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['epoch']==0 and state['optimizer'] is not None and state['train_args']['epochs']==3
    del state,model
    model=YOLO(str(checkpoint));epochs=[]
    model.add_callback('on_train_epoch_start',lambda trainer:epochs.append(trainer.epoch))
    model.train(resume=str(checkpoint),device='cpu',workers=0)
    with (folder/'native/results.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert epochs==[1,2],epochs
    assert len(rows)==3,len(rows)
    report={'fixture':'Synthetic rectangles, not personal photos or assignment metrics',
            'state':'passed','saved_epoch_index':0,'optimizer_state_present':True,
            'resumed_epoch_indices':epochs,'total_csv_epochs':len(rows),'imgsz':64,
            'note':'The real Lab 06 workflow retains the required 640 pixels and 100 epochs.'}
    (folder/'verification.json').write_text(json.dumps(report,indent=2))
    print(folder);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
