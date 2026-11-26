"""CSST106 laboratory workbench. Run with: streamlit run app.py"""
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'data'/'cache'/'matplotlib'))
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from core import *
import jobs

st.set_page_config(page_title='CSST106 · Vision laboratory',layout='wide')
st.markdown('''<style>
 .block-container {max-width:1320px;padding-top:2rem;padding-bottom:4rem}
 h1 {font-size:2.35rem!important;font-weight:650!important;letter-spacing:-.025em}
 h2 {font-size:1.5rem!important;margin-top:1.25rem!important}
 [data-testid="stSidebar"] {border-right:1px solid #dce5e1}
 [data-testid="stMetricValue"] {font-variant-numeric:tabular-nums}
 a {text-underline-offset:3px} ::selection {background:#c2e7d8}
 button:focus-visible,input:focus-visible {outline:3px solid #116d60!important;outline-offset:2px}
 </style>''',unsafe_allow_html=True)
ROOT.mkdir(exist_ok=True);RUNS.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)

TITLES=['Overview','01 · Image manipulation','02 · Sobel and Canny','03 · Histogram and contrast',
        '04 · Features and matching','05 · PCA eigenfaces','06 · YOLO11 detection','07 · Segmentation','Saved experiments']
with st.sidebar:
    st.title('Vision laboratory')
    st.caption('CSST106 · BSCS4B')
    page=st.radio('Laboratory navigation',TITLES,label_visibility='collapsed')
    st.divider();st.caption('Seven experiments · one local workspace')
    st.caption('Jeremy C. Reyes · 2026–2027')
    st.link_button('Class announcement','https://paios-classroom.com/classes/42')

def show_image(image,caption='',**kwargs):
    st.image(image,caption=caption,channels='BGR' if image.ndim==3 else 'RGB',**kwargs)

def input_image(key,default='Astronaut'):
    options=['Astronaut','Coffee','Coins','Camera']
    voc={}
    if key=='edge':
        from lab_inputs import EDGE_INPUT
        options.insert(0,'Packaged input.jpg')
    if key=='segment':
        voc={'VOC: '+p.stem:p for p in (DATA/'voc_subset').glob('val_*.png') if not p.stem.endswith('_truth')}
        options+=sorted(voc)
    a,b=st.columns([1,2])
    with a:choice=st.selectbox('Sample image',options,index=options.index(default),key=key+'_sample')
    with b:upload=st.file_uploader('Or upload an image',type=['png','jpg','jpeg'],key=key+'_upload')
    image=decode(upload.getvalue()) if upload else decode(EDGE_INPUT.read_bytes()) if key=='edge' and choice=='Packaged input.jpg' else decode(voc[choice].read_bytes()) if choice in voc else sample(choice)
    source_hash=hashlib.sha256(png(image)).hexdigest()
    hash_key=key+'_source_hash'
    if st.session_state.get(hash_key)!=source_hash:
        result_keys={'image':'image_result','edge':'edge_result','histogram':'hist_result','features':'feature_result','yolo':'yolo_result','segment':'seg_result'}
        st.session_state.pop(result_keys.get(key,''),None)
        st.session_state[hash_key]=source_hash
    st.session_state[key+'_voc_truth']=str(voc[choice].with_name(voc[choice].stem+'_truth.png')) if choice in voc and not upload else None
    source=('Uploaded: '+upload.name) if upload else 'Packaged input.jpg: disclosed scikit-image Camera replacement; not Jeremy\'s image' if choice=='Packaged input.jpg' else 'Public sample: '+choice+' · not a personal dataset photo'
    st.session_state[key+'_source_label']=source;st.caption(source)
    return image

def results(key,images,params,metrics=None,arrays=None):
    cols=st.columns(min(3,len(images)))
    for i,(name,a) in enumerate(images.items()):
        with cols[i%len(cols)]:
            show_image(a,name.replace('_',' ').capitalize(),width='stretch')
            st.download_button('Download '+name,png(a),name+'.png','image/png',key=key+'_'+name)
    if metrics:st.json(metrics,expanded=False)
    if st.button('Save experiment',key=key+'_save'):
        f=save_run(key,params,images,metrics,arrays);st.success('Saved '+f.name)
    with st.expander('Settings used'):st.json(params)

def histogram(channels):
    fig,axes=plt.subplots(1,len(channels),figsize=(4*len(channels),3),squeeze=False)
    for ax,(name,a) in zip(axes[0],channels.items()):
        ax.plot(np.bincount(a.ravel(),minlength=256),color='#116d60');ax.set(title=name,xlabel='Intensity',ylabel='Pixel count',xlim=(0,255))
    fig.tight_layout();st.pyplot(fig);plt.close(fig)

def observations(key,prompt):
    reference=ROOT/'examples'/key
    analysis=reference/'ANALYSIS.md'
    if analysis.exists():
        with st.expander('Completed reference analysis and figures'):
            st.markdown(analysis.read_text(encoding='utf-8'))
            figures=sorted(p for p in reference.rglob('*.png') if p.stem in ('comparison','histograms','seed_sensitivity','marker_sensitivity'))
            if figures:
                figure=st.selectbox('Reference figure',figures,format_func=lambda p:str(p.relative_to(reference)),key=key+'_reference_figure')
                st.image(str(figure),width='stretch')
                st.download_button('Download reference figure',figure.read_bytes(),figure.name,'image/png',key=key+'_reference_download')
    with st.expander('Observations and interpretation',expanded=True):
        st.write(prompt)
        file=DATA/(key+'_observations.txt')
        text=st.text_area('Your observations from these results',value=file.read_text(encoding='utf-8') if file.exists() else '',key=key+'_notes',height=120)
        if st.button('Save observations',key=key+'_notes_save'):file.write_text(text,encoding='utf-8');st.success('Observations saved.')

def job_panel(kind):
    paths=sorted(RUNS.glob('job_'+kind+'_*'),key=lambda p:p.stat().st_mtime,reverse=True)
    reference=ROOT/'examples'/kind
    if (reference/'result.json').exists():paths.append(reference)
    if not paths:st.info('No '+kind+' job has been run yet.');return
    selected=st.selectbox('Experiment job',paths,format_func=lambda p:'Completed reference experiment' if p==reference else p.name,key=kind+'_job')
    archived=selected==reference
    @st.fragment(run_every='3s')
    def progress():
        status={'state':'complete','message':'Saved sample outputs from the verified implementation run.'} if archived else jobs.status(selected)
        state_key='job_state_'+selected.name
        before=st.session_state.get(state_key);st.session_state[state_key]=status['state']
        if before is not None and before!=status['state'] and status['state'] in ('complete','failed','cancelled','paused'):
            st.rerun()
        st.write(status['state'].capitalize()+': '+status['message'])
        if status.get('epoch'):st.progress(min(status['epoch']/status.get('target_epochs',100),1))
        if status['state'] in ['starting','running'] and st.button('Cancel job',key=kind+'_cancel'):jobs.cancel(selected);st.rerun(scope='fragment')
        if kind=='yolo':
            params=json.loads((selected/'request.json').read_text())
            if params.get('mode','full')=='full' and status['state']=='running' and st.button('Pause after saved epoch',key='yolo_pause'):
                jobs.pause_yolo(selected);st.rerun(scope='fragment')
            if params.get('mode','full')=='full' and status['state'] in ('paused','cancelled','failed') and (selected/'progress.json').exists():
                if st.button('Resume original run',key='yolo_resume'):jobs.resume_yolo(selected);st.rerun()
        log=selected/'job.log'
        with st.expander('Worker log'):st.code(log.read_text(encoding='utf-8',errors='replace')[-10000:] if log.exists() else 'Starting…')
    progress()
    if (selected/'result.json').exists():
        r=json.loads((selected/'result.json').read_text());st.json(r,expanded=False)
        if kind=='pca':
            st.subheader('Measured recognition results');st.dataframe(pd.DataFrame(r['report']).T,width='stretch')
            for name in ['variance.png','eigenfaces.png','first_last.png','predictions.png','failures.png']:
                if (selected/name).exists():st.image(str(selected/name),caption=name.replace('_',' ').replace('.png',''),width='stretch')
            if (selected/'PCA_analysis.pdf').exists():st.download_button('Download PCA digital PDF',(selected/'PCA_analysis.pdf').read_bytes(),'PCA_analysis.pdf','application/pdf')
            with st.expander('Edit the PDF visual interpretation (new runs)'):
                note_file=selected/'visual_analysis.json';notes=json.loads(note_file.read_text()) if note_file.exists() else {}
                eigen_note=st.text_area('Eigenface interpretation',value=notes.get('eigenfaces',''),key=selected.name+'_eigen')
                failure_note=st.text_area('Image-specific failure hypotheses',value=notes.get('failure_analysis',''),key=selected.name+'_failure')
                if st.button('Update digital PDF',key=selected.name+'_pdf',disabled=archived):
                    note_file.write_text(json.dumps({'eigenfaces':eigen_note,'failure_analysis':failure_note},indent=2),encoding='utf-8')
                    from reporting import pca_pdf
                    pca_pdf(selected,r);st.rerun()
        elif kind=='semantic':
            cols=st.columns(2)
            for col,name in zip(cols,['classes.png','overlay.png']):
                with col:st.image(str(selected/name),width='stretch')
            st.write('Class legend',r['legend']);st.download_button('Download class-ID mask',(selected/'labels.png').read_bytes(),'labels.png','image/png')
        elif kind=='yolo':
            if r.get('mode')=='pilot':
                st.info(f"Pilot only. Estimated 100-epoch training time: {r['estimated_100_epoch_seconds']/3600:.2f} hours, plus startup/evaluation. Successful batch: {r['batch']}.")
            else:
                st.success('Custom weights: '+r['weights'])
                for p in list((selected/'training').glob('results.png'))+list((selected/'heldout').glob('confusion_matrix*.png')):st.image(str(p),width='stretch')
                if (selected/'ANALYSIS.md').exists():st.markdown((selected/'ANALYSIS.md').read_text(encoding='utf-8'))
                if st.button('Build private Lab 06 bundle'):
                    from lab06 import bundle
                    path=bundle(selected);st.download_button('Download private Lab 06 bundle',path.read_bytes(),path.name,'application/zip')
    return selected

@st.cache_resource
def yolo_model(path):
    from ultralytics import YOLO
    if path=='yolo11n.pt':
        dest=DATA/'models';dest.mkdir(parents=True,exist_ok=True)
        path=str(dest/'yolo11n.pt')
    return YOLO(path)

def inference(image,path,confidence):
    start=time.perf_counter();pred=yolo_model(path).predict(image,conf=confidence,verbose=False)[0]
    elapsed=time.perf_counter()-start
    rows=[]
    for box in pred.boxes:
        rows.append({'class':pred.names[int(box.cls[0])],'confidence':float(box.conf[0]),'xyxy':box.xyxy[0].tolist()})
    return pred.plot(),{'detections':rows,'inference_seconds':elapsed,'inference_fps':1/elapsed}

def annotate():
    from dataset import store_photo,save_annotation,build_dataset,validate_dataset,capture_progress,contact_sheets
    st.subheader('Curated bottle, cup and phone dataset')
    st.write('Public COCO collection: target 100 positive images per class plus 30 backgrounds. Mixed images can count toward several quotas. Import source boxes, visually review and correct them, then split whole source groups. These are not personally captured photographs.')
    with st.expander('Public curation, review and training guide'):
        st.markdown((ROOT/'LAB06_GUIDE.md').read_text(encoding='utf-8'))
    progress=capture_progress()
    if progress:st.dataframe(pd.DataFrame(progress).rename(columns={'session':'source/capture group'}),hide_index=True,width='stretch')
    session=st.text_input('Source/capture group for optional personal imports',placeholder='e.g. desk-daylight-01')
    photos=st.file_uploader('Optionally add your own photos',type=['jpg','jpeg','png'],accept_multiple_files=True)
    if st.button('Import photos'):
        if not photos:st.warning('Choose photos first.')
        else:
            for photo in photos:store_photo(photo.getvalue(),session)
            st.success('Photos imported. Review all objects and mark each image complete.')
    items=sorted((DATA/'photos').glob('*.png'))
    if items:
        selected=st.selectbox('Photo to annotate',items,format_func=lambda p:p.stem[:12])
        akey='corner_a_'+selected.stem;bkey='corner_b_'+selected.stem
        meta=json.loads(selected.with_suffix('.json').read_text());st.caption('Source/capture group: '+meta['session'])
        if meta.get('source_kind')=='coco_public':st.caption('Public COCO image '+str(meta['source']['image_id'])+'; existing source annotation requires visual review.')
        image=cv2.imread(str(selected));h,w=image.shape[:2]
        from streamlit_image_coordinates import streamlit_image_coordinates
        st.write('Click the image and use the buttons below to set opposite box corners. Coordinates are mapped back to original size. Add every object; a reviewed empty list means background-only.')
        scaled=cv2.resize(image,(min(w,850),round(h*min(w,850)/w)));sh,sw=scaled.shape[:2]
        point=streamlit_image_coordinates(cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB),key='annotate_'+selected.stem)
        c1,c2=st.columns(2)
        if c1.button('Use click as top-left') and point:st.session_state[akey]=(point['x']/sw,point['y']/sh)
        if c2.button('Use click as bottom-right') and point:st.session_state[bkey]=(point['x']/sw,point['y']/sh)
        label=st.selectbox('Object class',CLASSES)
        st.caption('Selected corners: '+str(st.session_state.get(akey))+' / '+str(st.session_state.get(bkey)))
        if st.button('Add box'):
            a=st.session_state.get(akey);b=st.session_state.get(bkey)
            if not a or not b:st.warning('Set both corners first.')
            else:
                x1,x2=sorted([a[0],b[0]]);y1,y2=sorted([a[1],b[1]])
                meta['boxes'].append([CLASSES.index(label),(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]);save_annotation(selected,meta['boxes'],False);st.session_state.pop(akey,None);st.session_state.pop(bkey,None);st.rerun()
        rows=pd.DataFrame(meta['boxes'],columns=['class_id','x_center','y_center','width','height'])
        edited=st.data_editor(rows,num_rows='dynamic',width='stretch',key='boxes_'+selected.stem)
        overlay=image.copy()
        for cid,x,y,bw,bh in meta['boxes']:
            cv2.rectangle(overlay,(int((x-bw/2)*w),int((y-bh/2)*h)),(int((x+bw/2)*w),int((y+bh/2)*h)),(0,180,80),2)
            cv2.putText(overlay,CLASSES[int(cid)],(int((x-bw/2)*w),max(15,int((y-bh/2)*h))),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,130,60),2)
        show_image(overlay,'Saved boxes',width='stretch')
        if st.button('Save boxes and mark reviewed'):
            save_annotation(selected,edited.values.tolist(),True);st.success('Annotation reviewed.')
        st.caption(f'{len(items)} photos imported; '+str(sum(json.loads(p.with_suffix('.json').read_text())['reviewed'] for p in items))+' reviewed')
    if st.button('Build group-separated train / val / test dataset'):st.json(build_dataset())
    if st.button('Validate existing dataset'):st.json(validate_dataset())
    if items and st.button('Generate private labelled contact sheets'):
        contact_sheets();st.success('Private review sheets saved under data/review.')
    sheets=sorted((DATA/'review').glob('contact_*.png'))
    if sheets:
        sheet=st.selectbox('Private annotation review sheet',sheets);st.image(str(sheet),width='stretch')
    if (DATA/'dataset/frozen.json').exists():
        st.caption('The exported dataset is frozen. Unfreezing permits corrections but changed data cannot resume previous runs.')
        if st.button('Unfreeze dataset for corrections'):
            from lab06 import unfreeze_dataset
            unfreeze_dataset();st.rerun()

try:
    if page=='Overview':
        st.title('Computer vision, made visible')
        st.write('Load an image. Inspect each transformation. Compare the evidence.')
        left,right=st.columns([1.15,1])
        with left:
            st.subheader('Seven laboratory experiments')
            rows=[('01','Image manipulation','Pixels, channels, geometry'),('02','Sobel and Canny','Derivatives and thin edges'),('03','Histogram and contrast','Global equalization, stretching, CLAHE'),('04','Features and matching','SURF, SIFT, ORB and upright mode'),('05','PCA eigenfaces','150-component recognition and analysis'),('06','YOLO11 detection','Pretrained inference and custom training'),('07','Segmentation','Thresholds, regions, edges, semantics')]
            st.dataframe(pd.DataFrame(rows,columns=['Lab','Experiment','Explore']),hide_index=True,width='stretch')
            st.write('Use the laboratory navigation to open an experiment. Every saved run retains its settings and outputs.')
        with right:
            im=sample('Coins');view,_=edges(im)
            show_image(np.hstack([gray(im),view['canny']]),'Public sample · grayscale pixels and Canny boundaries',width='stretch')
            st.caption('Actual processing output, not a mock visualization.')
        st.subheader('Readiness')
        st.write('Classical labs run with included samples. PCA downloads LFW on first use. DeepLab and YOLO download model weights. Custom YOLO training needs your annotated photos. SURF requires its isolated nonfree build.')
        st.link_button('Open setup and requirement checklist','https://paios-classroom.com/classes/42')
    elif page.startswith('01'):
        st.title('Image manipulation');st.write('Explore image representation, change a pixel, and transform the grid.')
        im=input_image('image');h,w=im.shape[:2];st.write({'height':h,'width':w,'channels':im.shape[2],'dtype':str(im.dtype),'OpenCV_channel_order':'BGR'})
        with st.form('manipulate'):
            a,b,c=st.columns(3)
            with a:
                row=st.number_input('Pixel row',0,h-1,h//2);col=st.number_input('Pixel column',0,w-1,w//2);colour=st.color_picker('New pixel colour','#ff4433')
            with b:
                x=st.number_input('Crop x',0,w-1,0);y=st.number_input('Crop y',0,h-1,0);rw=st.number_input('Crop width',1,w,max(1,w//2));rh=st.number_input('Crop height',1,h,max(1,h//2))
            with c:
                nw=st.number_input('Resize width',1,4096,max(1,w//2));nh=st.number_input('Resize height',1,4096,max(1,h//2));clockwise=st.checkbox('Rotate clockwise',True)
            run=st.form_submit_button('Apply image operations',type='primary')
        st.caption('Current pixel BGR: '+str(im[row,col].tolist())+' · Crop uses x/y origin and width/height.')
        if run:
            bgr=tuple(int(colour[i:i+2],16) for i in (5,3,1));params=dict(row=row,col=col,bgr=bgr,roi=[x,y,rw,rh],size=[nw,nh],clockwise=clockwise)
            st.session_state['image_result']=(manipulate(im,**params),params)
        if 'image_result' in st.session_state:
            out,params=st.session_state['image_result'];results('lab01',out,params)
            rr,cc=params['row'],params['col'];y0,y1=max(0,rr-5),min(out['original'].shape[0],rr+6);x0,x1=max(0,cc-5),min(out['original'].shape[1],cc+6)
            z1,z2=st.columns(2)
            with z1:show_image(cv2.resize(out['original'][y0:y1,x0:x1],(220,220),interpolation=cv2.INTER_NEAREST),'Before pixel edit · enlarged nearest-neighbor crop')
            with z2:show_image(cv2.resize(out['edited'][y0:y1,x0:x1],(220,220),interpolation=cv2.INTER_NEAREST),'After pixel edit · BGR '+str(params['bgr']))
        observations('lab01','An image stores a numeric value per channel at every grid position. Grayscale has one channel; OpenCV colour uses BGR. Describe the crop, changed pixel, resize and rotated dimensions you observe. Colour channels and pixel operations underpin photo editing, inspection and later vision pipelines.')
    elif page.startswith('02'):
        st.title('Sobel and Canny');im=input_image('edge','Packaged input.jpg')
        with st.form('edges'):
            a,b,c=st.columns(3);blur=a.select_slider('Gaussian kernel',[1,3,5,7,9],value=3);low=b.slider('Canny low',0,254,100);high=c.slider('Canny high',1,255,200)
            run=st.form_submit_button('Compare edge detectors',type='primary')
        st.caption('Lab baseline: blur 3×3, sigma 0; Sobel CV_64F, kernel 3; Canny 100/200. Set controls to these values to restore the baseline.')
        if run:
            out,arrays=edges(im,blur,low,high);st.session_state['edge_result']=(out,arrays,dict(blur=blur,low=low,high=high,source=st.session_state['edge_source_label']))
        if 'edge_result' in st.session_state:
            out,arrays,params=st.session_state['edge_result'];results('lab02',out,params,{'canny_edge_pixels':int(np.count_nonzero(out['canny']))},arrays)
        observations('lab02','Sobel returns directional intensity derivatives; combined magnitude can form broad bands and responds to noise. Canny adds thinning and hysteresis to produce a binary edge map. Compare the actual boundaries, continuity, fine detail and noisy regions. Raw signed gradients are saved separately from the normalized display.')
    elif page.startswith('03'):
        st.title('Histogram and contrast');im=input_image('histogram','Camera')
        a,b,c=st.columns(3);clip=a.slider('CLAHE clip limit',.1,10.,2.,.1);tiles=b.slider('CLAHE tiles per axis',2,16,8);colour=c.checkbox('Preserve colour using YUV luminance',False)
        if st.button('Compare contrast methods',type='primary'):
            out,channels=contrast(im,clip,tiles,colour);st.session_state['hist_result']=(out,channels,dict(clip=clip,tiles=tiles,colour=colour))
        if 'hist_result' in st.session_state:
            out,channels,params=st.session_state['hist_result'];results('lab03',out,params);histogram(channels)
            st.caption('Colour histograms describe the processed Y channel. Stretching maps the observed minimum/maximum to 0/255; constant images are unchanged.')
        observations('lab03','Histogram equalization remaps intensities using cumulative frequency to make crowded brightness levels more distinguishable. It is useful for low-contrast inspection and imaging. CLAHE works on local tiles, limits histogram peaks and interpolates between tiles, helping uneven illumination while limiting noise amplification. Compare detail, washed-out regions, noise and artifacts; more contrast is not always better.')
    elif page.startswith('04'):
        st.title('Features and matching');im=input_image('features')
        other=st.file_uploader('Second image (optional)',type=['jpg','jpeg','png'])
        rotated=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE);b=decode(other.getvalue()) if other else rotated
        pair_hash=hashlib.sha256(png(b)).hexdigest()
        if st.session_state.get('feature_second_hash')!=pair_hash:st.session_state.pop('feature_result',None);st.session_state['feature_second_hash']=pair_hash
        st.caption('Without a second upload, comparison uses an explicitly rotated copy of the same sample.')
        a1,a2=st.columns(2);threshold=a1.number_input('SURF Hessian threshold',1,10000,400);ratio=a2.slider('Two-neighbor ratio threshold',.1,.95,.75,.05)
        upright=st.checkbox('Upright SURF (disable orientation invariance)')
        if st.button('Compare SURF, SIFT and ORB',type='primary'):
            all_images={'input_a':im,'input_b':b};stats=[]
            for algo in ['ORB','SIFT']:
                matched,rich,metric=features(im,b,algo,ratio=ratio);all_images[algo+'_matches']=matched;all_images[algo+'_keypoints']=rich;stats.append(metric)
            worker_python=ROOT/'.surf-venv'/'Scripts'/'python.exe'
            if worker_python.exists():
                f=save_run('surf_input',dict(threshold=threshold,upright=upright,ratio=ratio),{'a':im,'b':b})
                from feature_runner import run_surf_comparison
                try:
                    compared,stats=run_surf_comparison(f,threshold,upright,ratio);all_images.update(compared)
                except RuntimeError as exc:st.warning(str(exc))
            else:st.warning('SURF has not been built. Run setup-surf.ps1; ORB and SIFT results below are not substitutes labeled as SURF.')
            st.session_state['feature_result']=(all_images,stats,dict(threshold=threshold,upright=upright,ratio=ratio))
        if 'feature_result' in st.session_state:
            out,stats,params=st.session_state['feature_result'];st.dataframe(pd.DataFrame(stats),hide_index=True,width='stretch');results('lab04',out,params,{'comparisons':stats})
        observations('lab04','Increasing the Hessian threshold generally reduces weak SURF keypoints. Descriptors encode local appearance for comparison across images. SURF/SIFT use L2 distance; ORB uses bitwise Hamming distance. Upright SURF avoids orientation estimation and loses rotation invariance. Discuss the measured timing and accepted matches on the same pair; match count alone does not prove accuracy.')
    elif page.startswith('05'):
        st.title('PCA eigenfaces');st.write('Recognize LFW identities using the required training-only PCA and balanced RBF SVM.')
        st.code('LFW: min_faces_per_person=70, resize=0.4\nSplit: test_size=0.2, random_state=42\nPCA: 150, randomized SVD, whiten=True, random_state=42\nSVM: kernel="rbf", class_weight="balanced"')
        if st.button('Run PCA and SVM experiment',type='primary'):
            jobs.launch('pca',{});st.success('Background job started. You may navigate to other labs.')
        job_panel('pca');observations('lab05','Inspect the first three and last three eigenfaces. Describe visible broad lighting/shape patterns and finer residual detail. For each misclassified test face, record a specific visual hypothesis such as pose, shadow or occlusion; do not state a cause as proven. The PDF provides measured results and conceptual analysis; these observations remain your interpretation.')
    elif page.startswith('06'):
        st.title('YOLO11 object detection')
        mode=st.tabs(['Inference','Curated dataset','Train and evaluate'])
        with mode[0]:
            from lab06 import completed_models
            models=['yolo11n.pt']+completed_models();model=st.selectbox('Model',models);conf=st.slider('Confidence',.05,.95,.25,.05)
            st.caption('Phone-recorded video requires no PC webcam. Jeremy must accept the disclosed substitution for the live demonstration.')
            im=input_image('yolo','Coffee')
            if st.button('Detect objects',type='primary'):
                with st.spinner('Loading model and detecting objects…'):out,m=inference(im,model,conf)
                st.session_state['yolo_result']=(im.copy(),out,m,dict(model=model,confidence=conf))
            if 'yolo_result' in st.session_state:
                original,out,m,params=st.session_state['yolo_result'];results('lab06',{'input':original,'detections':out},params,m)
            video=st.file_uploader('Video for detection',type=['mp4','avi','mov'])
            process_requested=st.button('Process video')
            if process_requested and video is None:st.warning('Choose a video file before processing.')
            if process_requested and video:
                from media import process_video
                f=save_run('lab06_video',dict(model=model,confidence=conf));path=f/('input'+Path(video.name).suffix);path.write_bytes(video.getvalue());preview=st.empty()
                def update_frame(count,frame):
                    if count%10==0:preview.image(frame,channels='BGR',caption=f'{count} frames processed')
                metric=process_video(path,f/'detections.mp4',lambda frame:inference(frame,model,conf),update_frame)
                (f/'video_metrics.json').write_text(json.dumps(metric));st.success('Saved video to '+str(f))
                if (f/'detections.mp4').exists():st.download_button('Download processed video',(f/'detections.mp4').read_bytes(),'detections.mp4','video/mp4')
            st.subheader('Local webcam')
            st.caption('Uses the camera attached to this Windows computer. Close other camera apps first.')
            col1,col2=st.columns(2)
            if col1.button('Start webcam'):st.session_state['camera_on']=True
            if col2.button('Stop webcam'):st.session_state['camera_on']=False
            if st.session_state.get('camera_on'):
                @st.fragment(run_every='.2s')
                def camera():
                    if 'capture' not in st.session_state:st.session_state['capture']=cv2.VideoCapture(0)
                    cap=st.session_state['capture'];ok,frame=cap.read()
                    if not ok:cap.release();st.session_state.pop('capture',None);st.session_state['camera_on']=False;st.error('Camera could not be read. Check Windows camera permission and close other apps.');return
                    rendered,metric=inference(frame,model,conf);st.image(rendered,channels='BGR',caption=f"Inference {metric['inference_seconds']*1000:.0f} ms · {metric['inference_fps']:.1f} inference FPS")
                camera()
            elif 'capture' in st.session_state:st.session_state.pop('capture').release()
        with mode[1]:annotate()
        with mode[2]:
            st.write('Run a three-epoch CPU pilot first. Full training starts fresh from YOLO11n for 100 total epochs at 640 px, seed 42, no early stopping. Pilot weights cannot be selected as final models. Validation selects best.pt; test evaluation runs once.')
            st.info('Training requires the reviewed curated dataset. This workflow uses the local CPU; read the pilot runtime estimate before the 100-epoch run.')
            if st.button('Validate dataset and run 3-epoch pilot'):
                jobs.launch('yolo',{'mode':'pilot'});st.success('Pilot started. The session split is now frozen.')
            if st.button('Validate dataset and start training',type='primary'):
                from dataset import validate_dataset
                st.json(validate_dataset());jobs.launch('yolo',{'mode':'full'});st.success('Full training started in background; it continues toward 100 total epochs.')
            job_panel('yolo')
        observations('lab06','Classification assigns a category to an entire image; detection locates each object and gives its category. A shelf image can be classified as containing drinks, while a detector boxes and labels individual bottles. Compare real precision, recall, missed objects and false alarms rather than judging only a successful webcam frame.')
    elif page.startswith('07'):
        st.title('Segmentation techniques');im=input_image('segment','Coins')
        classical,semantic=st.tabs(['Thresholds, regions and edges','Semantic classes'])
        with classical:
            st.write('Select a region-growing seed on the image; compare grouping by brightness, connectivity and boundaries.')
            from streamlit_image_coordinates import streamlit_image_coordinates
            disp=cv2.resize(im,(min(im.shape[1],850),round(im.shape[0]*min(im.shape[1],850)/im.shape[1])))
            click=streamlit_image_coordinates(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB),key='region_seed')
            seed=(int(click['y']*im.shape[0]/disp.shape[0]),int(click['x']*im.shape[1]/disp.shape[1])) if click else (im.shape[0]//2,im.shape[1]//2)
            st.caption('Seed (row, column): '+str(seed))
            a,b,c=st.columns(3);tol=a.slider('Seed intensity tolerance',0,255,15);block=b.select_slider('Adaptive block size',[3,9,15,25,51,81],value=25);const=c.slider('Adaptive constant C',-20,30,8)
            a,b,c=st.columns(3);frac=a.slider('Watershed marker fraction',.1,.9,.45,.05);inv=b.checkbox('Dark objects as foreground');conn=c.selectbox('Region connectivity',[4,8])
            if st.button('Compare segmentation methods',type='primary'):
                start=time.perf_counter();out,markers,m=segmentation(im,seed,tol,block,const,frac,inv,conn);m['total_seconds']=time.perf_counter()-start;st.session_state['seg_result']=(out,markers,m,dict(seed=seed,tolerance=tol,adaptive_block=block,adaptive_c=const,marker_fraction=frac,invert=inv,connectivity=conn))
            if 'seg_result' in st.session_state:
                out,markers,m,params=st.session_state['seg_result'];results('lab07',out,params,m,{'watershed_labels':markers})
                gt=st.file_uploader('Optional binary ground-truth mask (0/255)',type=['png'],key='binary_truth')
                if gt:
                    truth=np.asarray(Image.open(gt))
                    if truth.ndim!=2 or not set(np.unique(truth)).issubset({0,1,255}):raise ValueError('Choose a binary grayscale mask, not a colour visualization.')
                    rows=[]
                    for name in ['otsu','adaptive','region','edge_regions']:
                        score=mask_scores((out[name]>0).astype(int),(truth>0).astype(int),2,ignore=-1);rows.append({'method':name,'foreground_iou':score['classes'][1]['iou'],'foreground_dice':score['classes'][1]['dice']})
                    st.dataframe(rows)
            st.table(pd.DataFrame([['Thresholding','Fast and simple','Sensitive to lighting and overlapping intensities'],['Region-based','Uses similarity and connectivity','Seed leakage or watershed over-segmentation'],['Edge-based','Can localize clear boundaries','Noise and gaps break outlines'],['Semantic','Labels scene pixels by learned class','Needs model weights and matching class vocabulary']],columns=['Family','Advantage','Limitation']))
        with semantic:
            st.write('DeepLabV3-ResNet50 · pretrained COCO subset with 20 VOC classes plus background. No semantic training is required by this lab.')
            truth=st.file_uploader('Optional aligned VOC class-ID mask (0–20, void 255)',type=['png'],key='voc_truth')
            if st.button('Label every pixel with DeepLab',type='primary'):
                f=Path(jobs.launch('semantic',{}))
                # The worker waits for input readiness before inference.
                (f/'input.png').write_bytes(png(im))
                if truth:(f/'truth.png').write_bytes(truth.getvalue())
                elif st.session_state.get('segment_voc_truth'):(f/'truth.png').write_bytes(Path(st.session_state['segment_voc_truth']).read_bytes())
                (f/'input.ready').touch();st.success('Semantic job started; it remains active when changing pages.')
            job_panel('semantic')
        observations('lab07','Compare methods on the same input, then on uneven illumination, touching objects and a natural scene. Record seed sensitivity, over-segmentation, gaps, runtime and semantic label errors. Quantitative IoU is meaningful only with aligned, compatible ground truth. Unlabeled watershed regions are not semantic classes.')
    else:
        st.title('Saved experiments')
        folders=sorted([p for p in RUNS.iterdir() if p.is_dir()],key=lambda p:p.stat().st_mtime,reverse=True)
        if not folders:st.info('No experiments saved yet. Run a lab and choose Save experiment.')
        for folder in folders:
            with st.expander(folder.name):
                st.caption(str(folder));file=folder/'manifest.json'
                if file.exists():st.json(json.loads(file.read_text()),expanded=False)
                if (folder/'status.json').exists():st.json(jobs.status(folder))
                if st.button('Prepare experiment download',key='zip_'+folder.name):
                    buf=io.BytesIO()
                    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
                        for f in folder.rglob('*'):
                            if f.is_file() and f.suffix not in ['.pt','.pkl']:archive.write(f,f.relative_to(folder))
                    st.download_button('Download results ZIP',buf.getvalue(),folder.name+'.zip',key='download_'+folder.name)
except Exception as exc:
    st.error(str(exc))
    with st.expander('Technical details'):st.exception(exc)

# Release local camera when leaving its module, including full app reruns.
if not page.startswith('06') and 'capture' in st.session_state:
    st.session_state.pop('capture').release();st.session_state['camera_on']=False
