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
from feature_runner import run_surf_comparison

def main():
    p=argparse.ArgumentParser();p.add_argument('lab',type=int,choices=range(1,8));p.add_argument('--image');p.add_argument('--second');p.add_argument('--sample',choices=['Astronaut','Coffee','Coins','Camera'])
    p.add_argument('--hessian',type=int,default=400);p.add_argument('--upright',action='store_true');p.add_argument('--ratio',type=float,default=.75)
    p.add_argument('--pilot',action='store_true',help='Lab 06: time a separate three-epoch CPU pilot')
    p.add_argument('--resume',type=Path,help='Lab 06: resume this interrupted full-training job')
    a=p.parse_args()
    if (a.pilot or a.resume) and a.lab!=6:p.error('--pilot and --resume apply only to Lab 06')
    if a.hessian<1 or not 0<a.ratio<1:p.error('Hessian must be positive and ratio must be between 0 and 1.')
    if a.lab in [5,6]:
        if a.lab==6:
            from dataset import validate_dataset
            validate_dataset()
        if a.resume:
            if a.lab!=6 or a.pilot:p.error('--resume requires Lab 06 full training')
            print(jobs.resume_yolo(a.resume));return
        print(jobs.launch('pca' if a.lab==5 else 'yolo',{'mode':'pilot' if a.pilot else 'full'} if a.lab==6 else {}));return
    from lab_inputs import select_input
    im,source=select_input(a.lab,a.image,a.sample);h,w=im.shape[:2]
    params={'source':source};metrics={};arrays={}
    if a.lab==1:out=manipulate(im,row=h//2,col=w//2,bgr=(0,0,255),roi=(0,0,w//2,h//2),size=(max(1,w//2),max(1,h//2)))
    elif a.lab==2:out,arrays=edges(im);params.update(blur=3,low=100,high=200)
    elif a.lab==3:out,_=contrast(im);params.update(clip=2.,tiles=8,colour=False)
    elif a.lab==4:
        b=decode(Path(a.second).read_bytes()) if a.second else cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE);out={'input_a':im,'input_b':b};rows=[]
        # All detectors run in the same nonfree build, just as in the UI.
        folder=save_run('surf_input',params,{'a':im,'b':b})
        try:compared,rows=run_surf_comparison(folder,a.hessian,a.upright,a.ratio)
        except RuntimeError as exc:p.exit(1,str(exc)+'\n')
        out.update(compared);params.update(threshold=a.hessian,upright=a.upright,ratio=a.ratio)
        metrics={'comparisons':rows}
    else:out,markers,metrics=segmentation(im,(h//2,w//2));arrays={'watershed_labels':markers}
    print(save_run('lab'+str(a.lab).zfill(2),params,out,metrics,arrays))

if __name__=='__main__':main()
