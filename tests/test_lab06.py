"""Orchestration contracts use a fake trainer; no fabricated personal results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import json
import numpy as np
import pytest
import dataset
import lab06
from core import png


@pytest.fixture
def personal_fixture(tmp_path,monkeypatch):
    # Synthetic fixtures test workflow invariants only, never assignment accuracy.
    monkeypatch.setattr(dataset,'DATA',tmp_path)
    monkeypatch.setattr(lab06,'DATA',tmp_path)
    monkeypatch.setattr(lab06,'RUNS',tmp_path/'runs')
    (tmp_path/'runs').mkdir()
    for i in range(10):
        p=dataset.store_photo(png(np.full((24,24,3),i,np.uint8)),f'session-{i}')
        dataset.save_annotation(p,[[c,.5,.5,.2,.2] for c in range(3)])
    dataset.build_dataset()
    return tmp_path


class Factory:
    def __init__(self,folder,pause_at=None,oom=False,early_stop=False):
        self.folder=folder;self.pause_at=pause_at;self.oom=oom;self.early_stop=early_stop
        self.calls=[];self.vals=0
    def __call__(self,source):
        owner=self
        class Model:
            def __init__(self):self.callbacks={}
            def add_callback(self,event,callback):self.callbacks[event]=callback
            def train(self,**kw):
                owner.calls.append((source,kw))
                if owner.oom and kw.get('batch')==4:raise RuntimeError('out of memory')
                if kw.get('resume'):
                    previous=json.loads(Path(kw['resume']).read_text());start=previous['epoch']+1
                    target=previous['epochs'];batch=previous['batch'];name='training'
                    assert previous['optimizer']=='retained'
                else:start=0;target=kw['epochs'];batch=kw['batch'];name=kw['name']
                weights=owner.folder/name/'weights';weights.mkdir(parents=True,exist_ok=True)
                end=target-1 if owner.early_stop else target
                for epoch in range(start,end):
                    trainer=SimpleNamespace(epoch=epoch,args=SimpleNamespace(batch=batch),last=weights/'last.pt')
                    self.callbacks['on_train_epoch_start'](trainer)
                    trainer.last.write_text(json.dumps({'epoch':epoch,'epochs':target,'batch':batch,'optimizer':'retained'}))
                    (weights/'best.pt').write_bytes(b'fake trainer weights')
                    if owner.pause_at==epoch+1:(owner.folder/'pause.request').touch()
                    self.callbacks['on_model_save'](trainer)
            def val(self,**kw):
                owner.vals+=1
                assert kw['split']=='test'
                return SimpleNamespace(results_dict={'metrics/precision(B)':.5,'metrics/recall(B)':.25,'metrics/mAP50(B)':.4,'metrics/mAP50-95(B)':.2})
        return Model()


def test_pilot_oom_fallback_is_not_a_final_model(personal_fixture):
    folder=personal_fixture/'runs/job_yolo_pilot';folder.mkdir()
    factory=Factory(folder,oom=True)
    lab06.train_job(folder,lab06.prepare_job({'mode':'pilot'}),factory)
    result=lab06.read_json(folder/'result.json')
    assert result['epochs_completed']==3 and result['batch']==2
    assert result['estimated_100_epoch_seconds']>=0
    assert factory.vals==0 and not lab06.eligible_model(folder)
    assert all(kw['epochs']==3 for _,kw in factory.calls)


def test_pause_resume_retains_original_target_and_evaluates_once(personal_fixture):
    folder=personal_fixture/'runs/job_yolo_full';folder.mkdir()
    factory=Factory(folder,pause_at=1)
    lab06.train_job(folder,lab06.prepare_job({'mode':'full'}),factory)
    assert lab06.read_json(folder/'status.json')['state']=='paused'
    assert lab06.read_json(folder/'progress.json')['epochs_completed']==1
    assert not lab06.eligible_model(folder)
    # A process killed after writing the next file but before committing its
    # progress pointer must still be able to resume the previous valid pair.
    (folder/'checkpoints/epoch_002.pt').write_bytes(b'uncommitted checkpoint')
    assert lab06.resumable(folder)['mode']=='full'
    (folder/'pause.request').unlink();factory.pause_at=None
    lab06.train_job(folder,{'resume':True},factory)
    assert factory.calls[-1][1]['resume']==str(folder/'checkpoints/epoch_001.pt')
    assert lab06.read_json(folder/'progress.json')['epochs_completed']==100
    assert factory.vals==1 and lab06.eligible_model(folder)
    assert len(list((folder/'checkpoints').glob('*.pt')))==2
    lab06.evaluate_once(folder,folder/'training/weights/best.pt',factory)
    assert factory.vals==1
    with pytest.raises(ValueError,match='finished'):lab06.resumable(folder)


def test_dataset_change_blocks_resume_and_frozen_rebuild(personal_fixture):
    folder=personal_fixture/'runs/job_yolo_changed';folder.mkdir()
    lab06.train_job(folder,lab06.prepare_job({'mode':'full'}),Factory(folder,pause_at=1))
    with pytest.raises(ValueError,match='frozen'):dataset.build_dataset()
    label=next((personal_fixture/'dataset/labels/train').glob('*.txt'))
    label.write_text(label.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):lab06.resumable(folder)


def test_incomplete_full_training_never_evaluates(personal_fixture):
    folder=personal_fixture/'runs/job_yolo_short';folder.mkdir();factory=Factory(folder,early_stop=True)
    with pytest.raises(RuntimeError,match='99/100'):
        lab06.train_job(folder,lab06.prepare_job({'mode':'full'}),factory)
    assert factory.vals==0 and not (folder/'result.json').exists()


def test_interrupted_test_evaluation_is_not_silently_repeated(tmp_path):
    lab06.write_json(tmp_path/'evaluation.started.json',{'split':'test'})
    factory=Factory(tmp_path)
    with pytest.raises(ValueError,match='previously started'):lab06.evaluate_once(tmp_path,tmp_path/'best.pt',factory)
    assert factory.vals==0


def test_missing_personal_data_blocks_training_with_capture_instructions(tmp_path,monkeypatch):
    monkeypatch.setattr(dataset,'DATA',tmp_path)
    with pytest.raises(ValueError,match='Import and review'):lab06.prepare_job({'mode':'pilot'})


def test_annotation_nonfinite_and_fractional_classes_rejected(personal_fixture):
    photo=next((personal_fixture/'photos').glob('*.png'))
    for box in ([.5,.5,.5,.2,.2],[0,float('nan'),.5,.2,.2]):
        with pytest.raises(ValueError):dataset.save_annotation(photo,[box])
    raw=png(np.full((24,24,3),0,np.uint8))
    with pytest.raises(ValueError,match='another session'):dataset.store_photo(raw,'different')


def test_contact_sheets_and_public_summary_keep_private_material_local(personal_fixture,monkeypatch):
    sheets=dataset.contact_sheets()
    assert sheets and all(p.is_file() for p in sheets)
    assert dataset.capture_progress()[0]['reviewed']==1
    folder=personal_fixture/'runs/job_yolo_finished';folder.mkdir()
    lab06.train_job(folder,lab06.prepare_job({'mode':'full'}),Factory(folder))
    (folder/'demo').mkdir();(folder/'demo/review.txt').write_text('Private identifiable notes')
    lab06.write_json(folder/'demo/result.json',{'videos':{}})
    (personal_fixture/'examples').mkdir();monkeypatch.setattr(lab06,'ROOT',personal_fixture)
    public=lab06.publish_summary(folder)
    assert 'weights' not in lab06.read_json(public/'result.json')
    assert 'Private identifiable notes' not in (public/'ANALYSIS.md').read_text()
    assert not list(public.glob('*.pt')) and not list(public.glob('*.mp4'))
