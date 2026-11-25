"""Retrieve five VOC validation pairs from the documented HF parquet mirror.

Read original PNG bytes, NOT the dataset viewer's lossy JPEG mask previews.
Record mirror revision, original paths (when present), row IDs and hashes.
"""
import hashlib
import io
import json
import urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from core import DATA

REV='65505efaf0a3c4afe512dfc27c3abe2eac397bf9'
URL=f'https://huggingface.co/datasets/nateraw/pascal-voc-2012/resolve/{REV}/data/val-00000-of-00001.parquet'

def main():
    cache=DATA/'cache';cache.mkdir(parents=True,exist_ok=True)
    file=cache/'voc_val.parquet'
    if not file.exists():
        partial=file.with_suffix('.partial')
        with urllib.request.urlopen(URL,timeout=60) as response,partial.open('wb') as out:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                out.write(chunk)
        partial.replace(file)
    batch=next(pq.ParquetFile(file).iter_batches(batch_size=5));rows=batch.to_pylist()
    dest=DATA/'voc_subset';dest.mkdir(parents=True,exist_ok=True);manifest=[]
    for i,row in enumerate(rows):
        image=row['image'];mask=row['mask'];a=Image.open(io.BytesIO(image['bytes'])).convert('RGB');m=Image.open(io.BytesIO(mask['bytes']))
        labels=np.asarray(m)
        if labels.ndim==3:
            # The mirror stores original VOC colours as a lossless RGB PNG.
            # Invert the official VOC bit palette exactly; never nearest-match.
            palette=np.zeros((256,3),np.uint8)
            for label in range(256):
                value=label
                for bit in range(8):
                    for channel in range(3):palette[label,channel] |= ((value>>channel)&1)<<(7-bit)
                    value >>= 3
            packed=labels[:,:,0].astype(np.int32)*65536+labels[:,:,1].astype(np.int32)*256+labels[:,:,2]
            mapping={int(r)*65536+int(g)*256+int(b):i for i,(r,g,b) in enumerate(palette)}
            unknown=set(np.unique(packed))-set(mapping)
            if unknown:raise ValueError('Unexpected colours in mask; refusing lossy label conversion.')
            labels=np.vectorize(mapping.__getitem__,otypes=[np.uint8])(packed)
            m=Image.fromarray(labels)
        if labels.ndim!=2 or not set(np.unique(labels)).issubset(set(range(21))|{255}):raise ValueError('Mirror masks are not lossless VOC class IDs; refusing metrics.')
        if a.size!=m.size:raise ValueError('Image/mask mismatch.')
        stem='val_'+str(i).zfill(4);a.save(dest/(stem+'.png'));m.save(dest/(stem+'_truth.png'))
        manifest.append({'row':i,'image_path':image.get('path'),'mask_path':mask.get('path'),'stem':stem,'mask_sha256':hashlib.sha256(mask['bytes']).hexdigest()})
    (dest/'manifest.json').write_text(json.dumps({'source':URL,'revision':REV,'selection':'first five mirrored val rows; original lossless mask bytes','examples':manifest},indent=2),encoding='utf-8')
    print('Prepared',len(rows),'verified image/mask pairs in',dest)

if __name__=='__main__':main()
