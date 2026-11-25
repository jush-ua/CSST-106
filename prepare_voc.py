"""Select five official VOC2012 validation examples from an extracted dataset.

The source dataset's license/terms must accompany redistributed examples.
Usage: python prepare_voc.py --root C:\datasets\VOCdevkit\VOC2012
"""
import argparse
import json
import shutil
from pathlib import Path
from core import DATA

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    ids=(args.root/'ImageSets'/'Segmentation'/'val.txt').read_text().split()[:5]
    dest=DATA/'voc_subset';dest.mkdir(parents=True,exist_ok=True)
    for stem in ids:
        shutil.copyfile(args.root/'JPEGImages'/(stem+'.jpg'),dest/(stem+'.jpg'))
        shutil.copyfile(args.root/'SegmentationClass'/(stem+'.png'),dest/(stem+'_truth.png'))
    (dest/'manifest.json').write_text(json.dumps({'source':'PASCAL VOC2012 segmentation validation','ids':ids,'selection':'first five in official val.txt','root':str(args.root)},indent=2),encoding='utf-8')
    print(dest)

if __name__=='__main__':main()
