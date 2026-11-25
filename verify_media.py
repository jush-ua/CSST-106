"""Verify real video output and optional local webcam without saving camera data.

python verify_media.py --webcam
Public synthetic clip is explicitly a test fixture, never a personal dataset.
"""
import argparse
import json
import time
import cv2
from core import DATA,RUNS,sample,save_run
from media import process_video,webcam_probe

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--webcam',action='store_true');args=parser.parse_args()
    from ultralytics import YOLO
    model=YOLO(str(DATA/'models'/'yolo11n.pt'))
    def detect(frame):
        start=time.perf_counter();result=model.predict(frame,verbose=False,conf=.25)[0]
        return result.plot(),{'inference_seconds':time.perf_counter()-start,'detection_count':len(result.boxes)}
    folder=save_run('media_verification',{'fixture':'12 slightly rotated frames of public scikit-image Coffee (0 to 1.65 degrees)','webcam_requested':args.webcam})
    im=cv2.resize(sample('Coffee'),(480,320));source=folder/'fixture.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),12.,(480,320))
    if not writer.isOpened():raise RuntimeError('Fixture writer did not open.')
    try:
        for i in range(12):
            frame=cv2.warpAffine(im,cv2.getRotationMatrix2D((240,160),i*.15,1),(480,320),borderMode=cv2.BORDER_REFLECT)
            writer.write(frame)
    finally:writer.release()
    report={'video':process_video(source,folder/'detections.mp4',detect)}
    capture=cv2.VideoCapture(str(folder/'detections.mp4'));count=0;shapes=set();output_fps=capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok,frame=capture.read()
            if not ok:break
            count+=1;shapes.add(tuple(frame.shape))
    finally:capture.release()
    report['video'].update(decoded_output_frames=count,decoded_output_shapes=[list(x) for x in shapes],decoded_output_fps=output_fps,state='passed' if count==12 and shapes=={(320,480,3)} and abs(output_fps-12)<.01 else 'failed')
    if args.webcam:report['webcam']=webcam_probe(detect)
    (folder/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(folder);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
