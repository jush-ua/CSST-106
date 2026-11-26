"""Local Lab 06 training, resumable checkpoints, phone-video evidence and delivery.

python lab06.py demo --job runs/job_yolo_ID --video phone.mp4 --images new.jpg
python lab06.py bundle --job runs/job_yolo_ID
python lab06.py publish-summary --job runs/job_yolo_ID
"""
import argparse
import gc
import hashlib
import json
import shutil
import statistics
import time
import zipfile
from pathlib import Path

from core import ROOT, DATA, RUNS, CLASSES


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def write_json(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=float), encoding='utf-8', newline='\n')
    temporary.replace(path)


def eligible_model(folder):
    result=read_json(Path(folder)/'result.json', {})
    return (result.get('mode')=='full' and result.get('epochs_completed')==100
            and result.get('evaluation_complete') is True
            and (Path(folder)/'training/weights/best.pt').exists())


def completed_models():
    return [str(folder/'training/weights/best.pt') for folder in sorted(RUNS.glob('job_yolo_*')) if eligible_model(folder)]


def prepare_job(params):
    from dataset import freeze_dataset
    params=dict(params);mode=params.setdefault('mode','full')
    if mode not in ('pilot','full'):raise ValueError('Training mode must be pilot or full.')
    fingerprint,audit=freeze_dataset()
    params.update(dataset_fingerprint=fingerprint, dataset_audit=audit)
    # Match the batch that actually completed a pilot on this frozen dataset.
    pilots=[]
    for path in RUNS.glob('job_yolo_*/result.json'):
        r=read_json(path,{})
        if r.get('mode')=='pilot' and r.get('dataset_fingerprint')==fingerprint:pilots.append((path.stat().st_mtime,r))
    params['batch']=max(pilots,key=lambda row:row[0])[1]['batch'] if pilots and mode=='full' else 4
    return params


def verify_snapshot(folder):
    from dataset import dataset_fingerprint, validate_dataset
    snapshot=read_json(Path(folder)/'training_snapshot.json')
    if not snapshot:raise ValueError('No training snapshot found.')
    validate_dataset()
    if dataset_fingerprint()!=snapshot['dataset_fingerprint']:
        raise ValueError('Dataset changed since this run. Restore the frozen dataset or start a new experiment.')
    return snapshot


def resumable(folder):
    folder=Path(folder);snapshot=verify_snapshot(folder)
    if snapshot['mode']!='full':raise ValueError('Pilot checkpoints cannot resume a final model.')
    progress=read_json(folder/'progress.json',{})
    if (folder/'result.json').exists():raise ValueError('This full experiment has already finished.')
    epoch=progress.get('epochs_completed',0)
    if not 0<epoch<=100:raise ValueError('No completed epoch is available to resume.')
    if epoch<100:
        checkpoint=checkpoint_path(folder,progress)
        if not checkpoint.exists():raise ValueError('No resumable checkpoint exists.')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=progress.get('checkpoint_sha256'):
            raise ValueError('Checkpoint does not match its saved progress record.')
    return snapshot


def checkpoint_path(folder,progress=None):
    folder=Path(folder)
    progress=progress if progress is not None else read_json(folder/'progress.json',{})
    path=folder/progress.get('checkpoint_file','resume.pt')
    if not path.resolve().is_relative_to(folder.resolve()):raise ValueError('Checkpoint must remain in its experiment folder.')
    return path


def unfreeze_dataset():
    import jobs
    jobs.ensure_idle('yolo')
    path=DATA/'dataset/frozen.json'
    if path.exists():path.unlink()


class TrainingPaused(Exception):
    pass


def evaluate_once(folder, best, factory):
    """An evaluation interrupted midway is not silently repeated on held-out data."""
    folder=Path(folder);path=folder/'evaluation.json'
    if path.exists():return read_json(path)
    marker=folder/'evaluation.started.json'
    if marker.exists():
        raise ValueError('Held-out evaluation previously started without finishing. Inspect heldout outputs before an explicit recovery; it will not be silently repeated.')
    write_json(marker,{'started':time.time(),'split':'test'})
    model=factory(str(best))
    result=model.val(data=str(DATA/'dataset/data.yaml'),split='test',device='cpu',workers=0,
                     project=str(folder),name='heldout',plots=True)
    metrics=dict(result.results_dict)
    write_json(path,metrics)
    return metrics


def train_job(folder, params, factory=None):
    """Full means 100 total epochs. A pilot always starts separately from pretrained weights."""
    from dataset import dataset_fingerprint,validate_dataset
    from jobs import write_status
    folder=Path(folder)
    if factory is None:
        from ultralytics import YOLO
        factory=YOLO
    (DATA/'models').mkdir(parents=True,exist_ok=True)
    if params.get('resume'):
        snapshot=resumable(folder)
    else:
        if read_json(folder/'training_snapshot.json'):
            raise ValueError('Existing run must be explicitly resumed, never overwritten.')
        audit=validate_dataset();fingerprint=dataset_fingerprint()
        if fingerprint!=params.get('dataset_fingerprint'):raise ValueError('Dataset changed while this job was starting.')
        snapshot=dict(mode=params.get('mode','full'),dataset_fingerprint=fingerprint,dataset=audit,
                      target_epochs=3 if params.get('mode')=='pilot' else 100,imgsz=640,
                      batch=params.get('batch',4),seed=42,device='cpu',provenance=read_json(DATA/'dataset/provenance.json',{}))
        write_json(folder/'training_snapshot.json',snapshot)
    mode=snapshot['mode'];target=snapshot['target_epochs']
    progress=read_json(folder/'progress.json',{}) if params.get('resume') else {}
    durations=progress.get('epoch_seconds',[])
    completed=progress.get('epochs_completed',0)
    if mode=='full' and completed==100:
        best=folder/'training/weights/best.pt'
    else:
        batches=[params.get('resume_batch',progress.get('batch',snapshot['batch']))] if mode=='full' or params.get('resume') else [4,2,1]
        if batches[0] not in (1,2,4):raise ValueError('Batch must be 1, 2, or 4.')
        for batch in batches:
            if mode=='pilot':durations=[]
            name=f'pilot_b{batch}' if mode=='pilot' else 'training'
            resume_checkpoint=checkpoint_path(folder) if params.get('resume') else None
            model=factory(str(resume_checkpoint) if resume_checkpoint else str(DATA/'models/yolo11n.pt'))
            clock=[None];attempt_progress={}
            def epoch_start(trainer):clock[0]=time.perf_counter()
            def checkpoint_saved(trainer):
                epoch=int(trainer.epoch)+1
                elapsed=time.perf_counter()-clock[0] if clock[0] is not None else 0.
                durations.append(elapsed)
                attempt_progress.update(epochs_completed=epoch,target_epochs=target,epoch_seconds=durations[:],batch=int(trainer.args.batch))
                if mode=='full':
                    directory=folder/'checkpoints';directory.mkdir(exist_ok=True)
                    checkpoint=directory/f'epoch_{epoch:03d}.pt';temporary=checkpoint.with_suffix('.tmp')
                    shutil.copyfile(trainer.last,temporary);temporary.replace(checkpoint)
                    attempt_progress['checkpoint_file']=checkpoint.relative_to(folder).as_posix()
                    attempt_progress['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                write_json(folder/'progress.json',attempt_progress)
                if mode=='full':
                    # Keep the previous committed checkpoint until its replacement's
                    # progress pointer is durable. A killed writer leaves the old pair valid.
                    for old in directory.glob('epoch_*.pt'):
                        if int(old.stem.split('_')[-1])<epoch-1:old.unlink()
                write_status(folder,'running',f'{mode.title()}: {epoch}/{target} epochs saved.',epoch=epoch,target_epochs=target)
                # Interrupt only after the optimizer-bearing checkpoint has been copied.
                if (folder/'pause.request').exists():raise TrainingPaused()
            model.add_callback('on_train_epoch_start',epoch_start)
            model.add_callback('on_model_save',checkpoint_saved)
            write_status(folder,'running',f'{mode.title()} training on CPU, batch {batch}.',target_epochs=target)
            try:
                if params.get('resume'):
                    history=read_json(folder/'resume_history.json',[])
                    history.append(dict(from_epoch=completed,target_epochs=target,batch=batch,checkpoint=resume_checkpoint.name))
                    write_json(folder/'resume_history.json',history)
                    model.train(resume=str(resume_checkpoint),device='cpu',workers=0,batch=batch)
                else:
                    model.train(data=str(DATA/'dataset/data.yaml'),epochs=target,imgsz=640,batch=batch,seed=42,
                                patience=0,workers=0,device='cpu',save=True,save_period=-1,
                                project=str(folder),name=name,exist_ok=True)
                completed=attempt_progress.get('epochs_completed',0)
                if completed!=target:raise RuntimeError(f'Training ended at {completed}/{target}; not a completed experiment.')
                actual_batch=attempt_progress['batch']
                best=folder/name/'weights/best.pt'
                break
            except TrainingPaused:
                write_status(folder,'paused','Paused after a saved epoch. Resume this full run to continue toward 100 total epochs.')
                return
            except (RuntimeError,MemoryError) as exc:
                if mode!='pilot' or batch==1 or not isinstance(exc,MemoryError) and 'out of memory' not in str(exc).lower():raise
                write_status(folder,'running',f'Pilot batch {batch} ran out of memory; retrying with a smaller batch.')
            finally:
                del model;gc.collect()
        else:raise RuntimeError('No training attempt completed.')
    if mode=='pilot':
        mean=statistics.mean(durations[-2:])
        result=dict(mode='pilot',epochs_completed=3,batch=actual_batch,epoch_seconds=durations,
                    estimated_100_epoch_seconds=mean*100,dataset_fingerprint=snapshot['dataset_fingerprint'],
                    note='Estimate from the final two pilot epochs; startup and final evaluation add time. This is not a completed Lab 06 model.')
        write_json(folder/'result.json',result)
        write_status(folder,'complete','Three-epoch pilot complete. Read the runtime estimate before starting full training.')
        return
    if not best.exists():raise ValueError('Completed training has no best.pt weights.')
    verify_snapshot(folder)
    write_status(folder,'running','100 epochs complete; evaluating the held-out test split once.',epoch=100,target_epochs=100)
    metrics=evaluate_once(folder,best,factory)
    result=dict(mode='full',epochs_completed=100,evaluation_complete=True,test_metrics=metrics,
                weights=str(best),dataset=snapshot['dataset'],dataset_fingerprint=snapshot['dataset_fingerprint'],
                device='cpu',imgsz=640,seed=42,batch=read_json(folder/'progress.json')['batch'],provenance=snapshot.get('provenance',{}))
    write_json(folder/'result.json',result)
    write_report(folder)
    write_status(folder,'complete','100-epoch training and held-out evaluation complete. Add the separate new-image demonstration next.',epoch=100,target_epochs=100)


def write_report(folder):
    folder=Path(folder);r=read_json(folder/'result.json')
    if not r or r.get('mode')!='full':raise ValueError('A completed full run is required for the Lab 06 report.')
    demo=read_json(folder/'demo/result.json')
    notes=folder/'demo/review.txt'
    provenance=r.get('provenance',{})
    lines=['# Lab 06 Curated object detection experiment','',
           'Classes: 0=bottle, 1=cup, 2=phone. '
           f"Source kind: {provenance.get('kind','not recorded')}. Annotation source: {provenance.get('annotation_source','not recorded')}. "
           'Raw images, annotations, videos and weights remain local. Source/capture groups were split and frozen before training.','',
           provenance.get('disclosure','Image origin must be documented; no personal-photography claim is inferred.'),'',
           f"YOLO11n: {r['epochs_completed']} total epochs, input size 640, CPU, batch {r['batch']}, seed 42, no early stopping. "
           'Validation selected best.pt; the test split was evaluated once after training.','',
           '| Split | Images | Bottle instances | Cup instances | Phone instances |','|---|---:|---:|---:|---:|']
    for split,a in r['dataset'].items():
        c=a['instances'];lines.append(f"| {split} | {a['images']} | {c['bottle']} | {c['cup']} | {c['phone']} |")
    history=read_json(folder/'resume_history.json',[])
    if history:
        lines+=['','Resume history (the batch above is the final batch):']+[f"- After epoch {h['from_epoch']}, resumed toward {h['target_epochs']} total epochs with batch {h['batch']}." for h in history]
    lines+=['','## Held-out metrics','']+[f'- {key}: {value}' for key,value in r['test_metrics'].items()]
    lines+=['','See training/results.csv, training/results.png and heldout confusion matrices for losses, validation trends and class errors. '
            'No arbitrary accuracy target is asserted. Small test sets and limited physical object variety restrict generalization.','',
            '## Separate demonstration images and optional recorded video','',
            'The PC has no webcam. The public still-image demonstration is not a live-webcam test. Recorded video can be processed separately; live webcam testing requires a connected camera. '
            'Recorded playback FPS is not inference throughput. Unlabelled demonstration images/video have no precision or recall ground truth. '
            'The custom head uses local IDs 0/1/2; pretrained COCO has 80 output IDs and calls phone "cell phone". Demo predictions are compared qualitatively, not as a second held-out evaluation.','']
    if demo:
        lines.append(f"Separate still-image outputs: {len(demo.get('images',[]))} model/image comparisons.")
        for name,m in demo['videos'].items():
            lines.append(f"- {name}: {m['frames']} frames, {m['source_fps']:.2f} source FPS, {m['seconds']:.2f} s processing, "
                         f"{m['processing_fps']:.2f} processing FPS; decode check: {m['verification']}.")
        review=notes.read_text(encoding='utf-8').strip() if notes.exists() else ''
        lines+=['',review or 'Qualitative successes/errors still need review; add observations to demo/review.txt and regenerate the report.']
    else:lines+=['No separate demonstration outputs are saved for this run. Supply new images or an optional recorded video.']
    lines+=['','## Detection versus classification','',
            'Classification assigns a label to an entire image. Detection locates and labels individual objects: '
            'a desk photo may contain a bottle, cup and phone with separate boxes, instead of one overall scene label.','']
    (folder/'ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')


def demonstrate(folder, video=None, images=()):
    import cv2
    from ultralytics import YOLO
    from media import process_video
    from core import decode,png
    folder=Path(folder)
    if not eligible_model(folder):raise ValueError('Demonstration requires an evaluated 100-epoch custom model, not a pilot.')
    if video is None and not images:raise ValueError('Supply separate new images and/or a video.')
    training_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'dataset/images').rglob('*.png')}
    for path in images:
        if hashlib.sha256(png(decode(Path(path).read_bytes()))).hexdigest() in training_hashes:
            raise ValueError('Demonstration images must be separate from train, validation and test images.')
    out=folder/'demo';out.mkdir(exist_ok=True)
    records={'webcam_substitution':'Still-image or recorded-video inference; live webcam testing not performed.','videos':{},'images':[]}
    for name,weights in [('pretrained',DATA/'models/yolo11n.pt'),('custom',folder/'training/weights/best.pt')]:
        model=YOLO(str(weights))
        wanted=[cid for cid,label in model.names.items() if label in ('bottle','cup','cell phone','phone')]
        def predict(frame):
            r=model.predict(frame,conf=.25,verbose=False,classes=wanted)[0]
            return r.plot(),{'detection_count':len(r.boxes)}
        if video is not None:
            destination=out/(name+'.mp4');m=process_video(video,destination,predict)
            reader=cv2.VideoCapture(str(destination));count=0;fps=reader.get(cv2.CAP_PROP_FPS);shapes=set()
            try:
                while True:
                    ok,frame=reader.read()
                    if not ok:break
                    count+=1;shapes.add(tuple(frame.shape))
            finally:reader.release()
            m.update(decoded_frames=count,decoded_fps=fps,decoded_shapes=[list(s) for s in shapes],
                     processing_fps=m['frames']/m['seconds'],verification='passed' if count==m['frames'] and abs(fps-m['source_fps'])<.1 else 'failed')
            records['videos'][name]=m
        for index,path in enumerate(images):
            source=decode(Path(path).read_bytes());start=time.perf_counter();result=model.predict(source,conf=.25,verbose=False,classes=wanted)[0]
            image=out/f'{name}_new_{index+1:03d}.png';image.write_bytes(png(result.plot()))
            records['images'].append({'model':name,'input':Path(path).name,'output':image.name,'detections':len(result.boxes),'seconds':time.perf_counter()-start,
                                     'boxes':[{'class':result.names[int(b.cls[0])],'confidence':float(b.conf[0]),'xyxy':b.xyxy[0].tolist()} for b in result.boxes]})
    write_json(out/'result.json',records);write_report(folder)
    return out


def bundle(folder):
    folder=Path(folder)
    if not eligible_model(folder):raise ValueError('Complete full training/evaluation before creating a deliverable bundle.')
    write_report(folder);destination=folder/'Lab06-deliverables.zip'
    paths=[folder/'result.json',folder/'training_snapshot.json',folder/'ANALYSIS.md',folder/'training/weights/best.pt',folder/'training/results.csv',folder/'resume_history.json',folder/'evaluation.json',folder/'progress.json']
    for directory in ('training','heldout','demo'):
        paths+=list((folder/directory).glob('*.png'))
    paths+=list((folder/'demo').glob('*.mp4'))+list((folder/'demo').glob('*.json'))+list((folder/'demo').glob('*.txt'))
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.exists():archive.write(path,path.relative_to(folder))
    return destination


def publish_summary(folder):
    folder=Path(folder)
    if not eligible_model(folder):raise ValueError('Only completed, evaluated full runs can publish a numerical summary.')
    write_report(folder);destination=ROOT/'examples/lab06_custom';destination.mkdir(exist_ok=True)
    r=read_json(folder/'result.json');r.pop('weights',None)
    write_json(destination/'result.json',r)
    # Intentionally omit raw inputs, weights, contact sheets and user-written notes.
    text=(folder/'ANALYSIS.md').read_text(encoding='utf-8')
    notes=folder/'demo/review.txt'
    if notes.exists() and notes.read_text(encoding='utf-8').strip():text=text.replace(notes.read_text(encoding='utf-8').strip(),'Qualitative notes remain in the private local bundle.')
    text=text.replace('training/results.csv','results.csv').replace('training/results.png','results.png').replace('heldout confusion matrices','confusion_matrix.png and confusion_matrix_normalized.png')
    text+='\nSee DEMO_ANALYSIS.md for the published qualitative review and metric interpretation when available.\n'
    (destination/'ANALYSIS.md').write_text(text,encoding='utf-8',newline='\n')
    for source in [folder/'training/results.csv',folder/'training/results.png']+list((folder/'heldout').glob('confusion_matrix*.png')):
        if source.exists():shutil.copyfile(source,destination/source.name)
    if r.get('provenance',{}).get('kind')=='curated_public':
        from curate_coco import publish_provenance
        publish_provenance(destination)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['demo','bundle','report','publish-summary','unfreeze'])
    parser.add_argument('--job',type=Path);parser.add_argument('--video',type=Path);parser.add_argument('--images',nargs='*',type=Path,default=[])
    args=parser.parse_args()
    if args.action=='unfreeze':unfreeze_dataset();return
    if not args.job:parser.error('--job is required')
    if args.action=='demo':
        if not args.video and not args.images:parser.error('Supply --images and/or --video for the separate demonstration')
        print(demonstrate(args.job,args.video,args.images))
    elif args.action=='bundle':print(bundle(args.job))
    elif args.action=='publish-summary':print(publish_summary(args.job))
    else:write_report(args.job);print(args.job/'ANALYSIS.md')


if __name__=='__main__':main()
