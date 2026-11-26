"""Pure image-processing functions shared by the Web UI, scripts and tests."""
from __future__ import annotations
import hashlib
import json
import platform
import time
import uuid
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import cv2
import numpy as np
import os

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'
DATA = ROOT / 'data'
CLASSES = ['bottle', 'mug', 'phone']
os.environ.setdefault('TORCH_HOME',str(DATA/'models'/'torch'))
os.environ.setdefault('YOLO_CONFIG_DIR',str(DATA/'cache'/'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(DATA/'cache'/'matplotlib'))
(DATA/'cache'/'matplotlib').mkdir(parents=True,exist_ok=True)
cv2.ocl.setUseOpenCL(False)

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

def decode(raw):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('This file could not be read as an image. Choose a PNG or JPEG.')
    if image.shape[0] * image.shape[1] > 25_000_000:
        raise ValueError('Image exceeds 25 million pixels. Resize it before uploading.')
    return image

def png(image):
    ok, buf = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('PNG encoding failed.')
    return buf.tobytes()

def display_u8(image):
    if image.dtype == np.uint8:
        return image
    a = np.nan_to_num(image.astype(float))
    return cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def sample(name):
    """Public scikit-image sample images, never presented as personal photos."""
    from skimage import data
    images = {'Astronaut': data.astronaut, 'Coffee': data.coffee,
              'Coins': data.coins, 'Camera': data.camera}
    a = images[name]()
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if a.ndim == 3 else cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

def save_run(lab, params, images=None, metrics=None, arrays=None):
    directory = RUNS / f'{lab}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}'
    directory.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ['numpy','opencv-python','scikit-learn','torch','torchvision','ultralytics','streamlit']:
        try: versions[package] = version(package)
        except PackageNotFoundError: pass
    manifest = {'lab': lab, 'parameters': params, 'metrics': metrics or {},
                'versions': versions, 'python': platform.python_version(), 'time': time.time(),
                'images': {}, 'arrays': []}
    for name, image in (images or {}).items():
        file = directory / f'{name}.png'
        file.write_bytes(png(image))
        manifest['images'][name] = {'file': file.name, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}
    for name, a in (arrays or {}).items():
        np.save(directory / f'{name}.npy', a)
        manifest['arrays'].append(f'{name}.npy')
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    return directory

def manipulate(image, *, row, col, bgr, roi, size, clockwise=True):
    h, w = image.shape[:2]
    x, y, rw, rh = map(int, roi)
    if not (0 <= row < h and 0 <= col < w): raise ValueError('Pixel is outside the image.')
    if min(rw, rh) <= 0 or x < 0 or y < 0 or x+rw > w or y+rh > h:
        raise ValueError('Crop must lie entirely inside the image.')
    if min(size) < 1 or max(size) > 4096: raise ValueError('Resize dimensions must be 1–4096.')
    edited = image.copy()
    edited[row, col] = bgr
    return {'original': image, 'grayscale': gray(image), 'edited': edited,
            'crop': image[y:y+rh,x:x+rw].copy(), 'resized': cv2.resize(image, tuple(map(int,size))),
            'rotated': cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE)}

def edges(image, blur=3, low=100, high=200):
    if blur < 1 or blur % 2 == 0: raise ValueError('Blur kernel must be positive and odd.')
    if not 0 <= low < high <= 255: raise ValueError('Canny low must be below high, both 0–255.')
    g = gray(image)
    smooth = cv2.GaussianBlur(g, (blur,blur), 0)
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx,gy)
    # First three entries form the exact original / Sobel / Canny UI row.
    return {'original':g, 'sobel':display_u8(mag),
            'canny':cv2.Canny(smooth,low,high), 'blurred':smooth}, {'gx':gx,'gy':gy,'magnitude':mag}

def stretch(a):
    lo, hi = float(a.min()), float(a.max())
    return a.copy() if hi == lo else np.rint((a.astype(float)-lo)*255/(hi-lo)).clip(0,255).astype(np.uint8)

def contrast(image, clip=2.0, tiles=8, colour=False):
    clahe = cv2.createCLAHE(clipLimit=float(clip),tileGridSize=(tiles,tiles))
    channel = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)[:,:,0] if colour else gray(image)
    versions = {'original':channel,'stretched':stretch(channel),
                'equalized':cv2.equalizeHist(channel),'clahe':clahe.apply(channel)}
    if colour:
        base = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        out = {}
        for name,a in versions.items():
            temp=base.copy();temp[:,:,0]=a
            out[name]=image if name=='original' else cv2.cvtColor(temp,cv2.COLOR_YUV2BGR)
        return out, versions
    return versions, versions

def features(a,b,algorithm='ORB',threshold=400,upright=False,ratio=.75):
    if algorithm == 'ORB': detector=cv2.ORB_create(nfeatures=1500);norm=cv2.NORM_HAMMING
    elif algorithm == 'SIFT': detector=cv2.SIFT_create(nfeatures=1500);norm=cv2.NORM_L2
    elif algorithm == 'SURF':
        detector=cv2.xfeatures2d.SURF_create(hessianThreshold=threshold,upright=upright);norm=cv2.NORM_L2
    else: raise ValueError('Unknown detector')
    start=time.perf_counter();ka,da=detector.detectAndCompute(gray(a),None);kb,db=detector.detectAndCompute(gray(b),None)
    detect_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();good=[]
    if da is not None and db is not None and len(db)>=2:
        for pair in cv2.BFMatcher(norm).knnMatch(da,db,k=2):
            if len(pair)==2 and pair[0].distance < ratio*pair[1].distance:good.append(pair[0])
    match_ms=(time.perf_counter()-start)*1000
    output=cv2.drawMatches(a,ka,b,kb,good[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    rich=cv2.drawKeypoints(a,ka,None,(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    stats={'algorithm':algorithm,'keypoints_a':len(ka),'keypoints_b':len(kb),
           'descriptor_dimensions':None if da is None else int(da.shape[1]),
           'detect_describe_ms':detect_ms,'match_ms':match_ms,'accepted_matches':len(good)}
    return output,rich,stats

def region_grow(image, seed, tolerance=15, connectivity=4):
    from collections import deque
    a=gray(image).astype(float);h,w=a.shape;r,c=map(int,seed)
    if not (0<=r<h and 0<=c<w):raise ValueError('Seed is outside the image.')
    mask=np.zeros((h,w),bool);visited=mask.copy();queue=deque([(r,c)]);value=a[r,c]
    dirs=[(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity==8:dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    while queue:
        y,x=queue.popleft()
        if visited[y,x]:continue
        visited[y,x]=True
        if abs(a[y,x]-value)>tolerance:continue
        mask[y,x]=True
        for dy,dx in dirs:
            yy,xx=y+dy,x+dx
            if 0<=yy<h and 0<=xx<w and not visited[yy,xx]:queue.append((yy,xx))
    return mask

def segmentation(image, seed, tolerance=15, adaptive_block=25, adaptive_c=8, marker_fraction=.45, invert=False, connectivity=4):
    start=time.perf_counter();timings={}
    g=gray(image);blur=cv2.GaussianBlur(g,(3,3),0)
    typ=cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    timings['preprocess']=(time.perf_counter()-start)*1000;stage=time.perf_counter()
    t,otsu=cv2.threshold(blur,0,255,typ+cv2.THRESH_OTSU)
    timings['otsu']=(time.perf_counter()-stage)*1000;stage=time.perf_counter()
    adaptive=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,typ,adaptive_block,adaptive_c)
    timings['adaptive']=(time.perf_counter()-stage)*1000;stage=time.perf_counter()
    grown=region_grow(g,seed,tolerance,connectivity).astype(np.uint8)*255
    timings['region']=(time.perf_counter()-stage)*1000;stage=time.perf_counter()
    kernel=np.ones((3,3),np.uint8);clean=cv2.morphologyEx(otsu,cv2.MORPH_OPEN,kernel)
    dist=cv2.distanceTransform(clean,cv2.DIST_L2,5)
    sure_fg=(dist>marker_fraction*dist.max()).astype(np.uint8)*255 if dist.max()>0 else np.zeros_like(clean)
    sure_bg=cv2.dilate(clean,kernel,iterations=2)
    n,markers=cv2.connectedComponents(sure_fg);markers=markers+1
    markers[cv2.subtract(sure_bg,sure_fg)>0]=0
    markers=cv2.watershed(image.copy(),markers)
    palette=np.random.default_rng(42).integers(40,240,(max(n+1,2),3),dtype=np.uint8)
    water=palette[np.clip(markers,0,len(palette)-1)];water[markers<=1]=0;water[markers==-1]=(0,0,255)
    timings['watershed']=(time.perf_counter()-stage)*1000;stage=time.perf_counter()
    canny=cv2.Canny(blur,100,200);linked=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)
    contours,_=cv2.findContours(linked,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    filled=np.zeros_like(g);cv2.drawContours(filled,contours,-1,255,cv2.FILLED)
    timings['edge_regions']=(time.perf_counter()-stage)*1000
    timings['total']=(time.perf_counter()-start)*1000
    return {'original':image,'otsu':otsu,'adaptive':adaptive,'region':grown,'watershed':water,'edge_regions':filled,'canny_edges':canny}, markers, {'otsu_threshold':t,'watershed_seed_components':n-1,'closed_contours':len(contours),'timings_ms':timings}

def mask_scores(pred,truth,num_classes,ignore=255):
    if pred.shape!=truth.shape:raise ValueError('Prediction and truth shapes must match; masks cannot be silently resized.')
    valid=truth!=ignore
    if not valid.any():return {'pixel_accuracy':None,'mean_iou':None,'classes':[]}
    rows=[]
    for c in range(num_classes):
        p=(pred==c)&valid;t=(truth==c)&valid;inter=int((p&t).sum());union=int((p|t).sum());den=int(p.sum()+t.sum())
        rows.append({'class':c,'iou':inter/union if union else None,'dice':2*inter/den if den else None})
    ious=[r['iou'] for r in rows if r['iou'] is not None]
    return {'pixel_accuracy':float((pred[valid]==truth[valid]).mean()),'mean_iou':float(np.mean(ious)) if ious else None,'classes':rows}
