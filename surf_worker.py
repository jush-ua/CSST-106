"""Run only in .surf-venv, which has the explicitly built nonfree OpenCV."""
import json
import sys
from pathlib import Path
import cv2
from core import features

if __name__=='__main__':
    folder=Path(sys.argv[1]);threshold=int(sys.argv[2]);upright=bool(int(sys.argv[3]));ratio=float(sys.argv[4])
    a=cv2.imread(str(folder/'a.png'));b=cv2.imread(str(folder/'b.png'))
    records=[]
    for th in sorted(set([100,400,1000,threshold])):
        for up in [False,True]:
            matches,rich,stats=features(a,b,'SURF',th,up,ratio)
            stats.update(hessian_threshold=th,upright=up);records.append(stats)
            if th==threshold and up==upright:
                cv2.imwrite(str(folder/'SURF_matches.png'),matches);cv2.imwrite(str(folder/'SURF_keypoints.png'),rich)
    (folder/'surf.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
