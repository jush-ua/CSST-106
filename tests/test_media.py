import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pytest
from media import process_video

def test_processed_video_decodes_every_frame_at_original_shape(tmp_path):
    source=tmp_path/'input.avi';dest=tmp_path/'output.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    try:
        for i in range(4):writer.write(np.full((48,64,3),i*40,np.uint8))
    finally:writer.release()
    report=process_video(source,dest,lambda a:(a,{}));assert report['frames']==4
    reader=cv2.VideoCapture(str(dest));decoded=[]
    try:
        while True:
            ok,frame=reader.read()
            if not ok:break
            decoded.append(frame.shape)
    finally:reader.release()
    assert decoded==[(48,64,3)]*4

def test_output_cannot_replace_input(tmp_path):
    p=tmp_path/'same.mp4'
    with pytest.raises(ValueError,match='overwrite'):process_video(p,p,lambda a:(a,{}))

def test_unreadable_video_reports_problem(tmp_path):
    with pytest.raises(ValueError,match='opened'):process_video(tmp_path/'missing.mp4',tmp_path/'out.mp4',lambda a:(a,{}))
