"""Disk-backed jobs survive Streamlit page changes. No training in UI reruns."""
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from core import ROOT, RUNS

def launch(kind, params):
    import psutil
    for previous in RUNS.glob('job_'+kind+'_*'):
        if status(previous)['state'] in ('starting','running'):
            try:
                info=json.loads((previous/'pid.json').read_text());cmd=psutil.Process(info['pid']).cmdline()
                if str(previous.resolve()).lower() in [str(x).lower() for x in cmd]:
                    raise ValueError('A '+kind+' job is already running. Cancel it or wait before starting another.')
            except (psutil.NoSuchProcess,FileNotFoundError,KeyError,json.JSONDecodeError):pass
    folder=RUNS / ('job_'+kind+'_'+uuid.uuid4().hex[:10]);folder.mkdir(parents=True,exist_ok=True)
    (folder/'request.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (folder/'job.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'worker.py'),kind,str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
    (folder/'pid.json').write_text(json.dumps({'pid':process.pid,'started':time.time(),'kind':kind}),encoding='utf-8')
    return str(folder)

def status(folder):
    f=Path(folder)
    file=f/'status.json'
    try:return json.loads(file.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return {'state':'starting','message':'Starting worker…'}

def cancel(folder):
    f=Path(folder);(f/'cancel.request').touch()
    pid=json.loads((f/'pid.json').read_text())['pid']
    if status(f)['state'] not in ('complete','failed','cancelled'):
        # Never kill an unrelated process if an old PID has been reused.
        import psutil
        try:
            command=psutil.Process(pid).cmdline()
            if not any(str(f.resolve()).lower()==str(x).lower() for x in command):
                raise ValueError('Job PID no longer belongs to this experiment; not terminating it.')
        except psutil.NoSuchProcess:
            write_status(f,'cancelled','Worker is no longer running.');return
        if os.name=='nt':
            stopped=subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
            if stopped.returncode and psutil.pid_exists(pid):raise ValueError('Windows could not stop this worker. Its status has not been changed.')
        else:os.kill(pid,signal.SIGTERM)
        write_status(f,'cancelled','Cancelled by user; completed artifacts retained.')

def write_status(folder,state,message,**extra):
    f=Path(folder);tmp=f/'status.tmp'
    tmp.write_text(json.dumps({'state':state,'message':message,**extra},indent=2,default=str),encoding='utf-8')
    tmp.replace(f/'status.json')
