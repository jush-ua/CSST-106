"""Video processing used by both the Streamlit UI and hardware verification."""
from pathlib import Path
import time
import cv2

def process_video(source, destination, predict_frame, progress=None):
    """predict_frame returns (annotated BGR frame, metrics). Always release I/O."""
    if Path(source).resolve()==Path(destination).resolve():raise ValueError('Video output must not overwrite its input.')
    capture=cv2.VideoCapture(str(source));writer=None;count=0;start=time.perf_counter()
    if not capture.isOpened():capture.release();raise ValueError('Video could not be opened. Choose a supported MP4, AVI or MOV.')
    fps=capture.get(cv2.CAP_PROP_FPS)
    if fps<=0:fps=30.
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            annotated,_=predict_frame(frame)
            if annotated.shape!=frame.shape:raise ValueError('Detector changed the video frame shape.')
            if writer is None:
                writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                if not writer.isOpened():raise ValueError('MP4 output encoder is unavailable.')
            writer.write(annotated);count+=1
            if progress:progress(count,annotated)
    finally:
        capture.release()
        if writer is not None:writer.release()
    if count==0:raise ValueError('Video contains no readable frames.')
    return {'frames':count,'seconds':time.perf_counter()-start,'source_fps':fps,'output':Path(destination).name}

def webcam_probe(predict_frame, camera_index=0, frames=10):
    """Read a short live burst, run inference once and verify release/reopen.

    No frame bytes are written to disk. Only non-image measurements return.
    """
    cap=cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return {'state':'unavailable','camera_index':camera_index,'message':'Camera could not open. Check device connection, Windows privacy settings and other camera apps.'}
    count=0;shape=None;start=time.perf_counter();inference_metrics=None
    try:
        for i in range(frames):
            ok,frame=cap.read()
            if not ok:break
            count+=1;shape=list(frame.shape)
            if i==frames-1:
                _,stats=predict_frame(frame)
                # Keep timing and counts, not identities/scene content.
                inference_metrics={k:v for k,v in stats.items() if k in ('inference_seconds','detection_count')}
    finally:cap.release()
    probe=cv2.VideoCapture(camera_index)
    try:reopened=probe.isOpened()
    finally:probe.release()
    return {'state':'passed' if count==frames and reopened and inference_metrics is not None else 'failed',
            'frames_read':count,'requested_frames':frames,'frame_shape':shape,'elapsed_seconds':time.perf_counter()-start,
            'camera_reopened_after_release':reopened,'inference':inference_metrics,'saved_camera_frames':0}
