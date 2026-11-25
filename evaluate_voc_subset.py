"""Reproduce semantic inference/IoU on the five saved validation samples."""
from pathlib import Path
import json
import uuid
from core import DATA,RUNS
from worker import semantic_job

if __name__=='__main__':
    summary=[]
    for image in sorted((DATA/'voc_subset').glob('val_*.png')):
        if image.stem.endswith('_truth'):continue
        f=RUNS/('job_semantic_'+uuid.uuid4().hex[:10]);f.mkdir(parents=True)
        (f/'request.json').write_text(json.dumps({'source':image.name}));(f/'input.png').write_bytes(image.read_bytes())
        (f/'truth.png').write_bytes(image.with_name(image.stem+'_truth.png').read_bytes());(f/'input.ready').touch()
        semantic_job(f,{'source':image.name});r=json.loads((f/'result.json').read_text())
        summary.append({'sample':image.stem,'job':f.name,'pixel_accuracy':r['scores']['pixel_accuracy'],'mean_iou':r['scores']['mean_iou'],'inference_seconds':r['inference_seconds']})
    if not summary:raise SystemExit('No VOC pairs. Run fetch_voc_subset.py first.')
    dest=RUNS/'voc_subset_evaluation.json';dest.write_text(json.dumps(summary,indent=2));print(dest)
