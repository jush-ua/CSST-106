"""Commented standalone entry point for reproducible classical experiments.

Examples:
  python run_lab.py 1 --image photo.jpg
  python run_lab.py 2 --sample Camera
  python run_lab.py 3 --image photo.jpg
  python run_lab.py 4 --image a.jpg --second b.jpg
  python run_lab.py 5
  python run_lab.py 6                 # validates personal dataset, then trains
  python run_lab.py 7 --sample Coins
"""
import argparse
from core import *
import jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('lab',type=int,choices=range(1,8));p.add_argument('--image');p.add_argument('--second');p.add_argument('--sample',default='Astronaut',choices=['Astronaut','Coffee','Coins','Camera']);a=p.parse_args()
    if a.lab in [5,6]:
        if a.lab==6:
            from dataset import validate_dataset
            validate_dataset()
        print(jobs.launch('pca' if a.lab==5 else 'yolo',{}));return
    im=decode(Path(a.image).read_bytes()) if a.image else sample(a.sample);h,w=im.shape[:2]
    params={'source':a.image or 'scikit-image:'+a.sample};metrics={};arrays={}
    if a.lab==1:out=manipulate(im,row=h//2,col=w//2,bgr=(0,0,255),roi=(0,0,w//2,h//2),size=(max(1,w//2),max(1,h//2)))
    elif a.lab==2:out,arrays=edges(im);params.update(blur=3,low=100,high=200)
    elif a.lab==3:out,_=contrast(im);params.update(clip=2.,tiles=8,colour=False)
    elif a.lab==4:
        b=decode(Path(a.second).read_bytes()) if a.second else cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE);out={'input_a':im,'input_b':b};rows=[]
        for algo in ['SIFT','ORB']:
            matched,rich,stats=features(im,b,algo);out[algo+'_matches']=matched;out[algo+'_keypoints']=rich;rows.append(stats)
        metrics={'comparisons':rows,'SURF':'Run isolated surf_worker after setup-surf.ps1'}
    else:out,markers,metrics=segmentation(im,(h//2,w//2));arrays={'watershed_labels':markers}
    print(save_run('lab'+str(a.lab).zfill(2),params,out,metrics,arrays))

if __name__=='__main__':main()
