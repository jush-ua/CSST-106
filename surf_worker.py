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
    # Compare all algorithms in the same OpenCV build and process.
    for algorithm in ['SIFT','ORB']:
        matches,rich,stats=features(a,b,algorithm,ratio=ratio)
        stats.update(opencv_version=cv2.__version__)
        cv2.imwrite(str(folder/(algorithm+'_matches.png')),matches)
        cv2.imwrite(str(folder/(algorithm+'_keypoints.png')),rich)
        records.append(stats)
    for th in sorted(set([100,400,1000,threshold])):
        for up in [False,True]:
            matches,rich,stats=features(a,b,'SURF',th,up,ratio)
            stats.update(hessian_threshold=th,upright=up,opencv_version=cv2.__version__);records.append(stats)
            if th==threshold and up==upright:
                cv2.imwrite(str(folder/'SURF_matches.png'),matches);cv2.imwrite(str(folder/'SURF_keypoints.png'),rich)
            if th==threshold and up:
                cv2.imwrite(str(folder/'U_SURF_matches.png'),matches)
    (folder/'surf.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
