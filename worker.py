"""Standalone reproducible tasks: python worker.py pca|yolo|semantic JOB_FOLDER."""
import json
import os
import sys
import time
import traceback
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
from core import DATA,ROOT
from jobs import write_status
os.environ.setdefault('MPLCONFIGDIR',str(DATA/'cache'/'matplotlib'))

def pca_job(folder,params):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report,confusion_matrix
    import joblib
    start=time.perf_counter();write_status(folder,'running','Fetching LFW; the first run downloads the dataset.')
    lfw=fetch_lfw_people(min_faces_per_person=70,resize=.4,data_home=DATA/'cache'/'lfw')
    X,y=lfw.data,lfw.target;h,w=lfw.images.shape[1:]
    indices=np.arange(len(X));train,test=train_test_split(indices,test_size=.2,random_state=42)
    if min(X[train].shape)<150:raise ValueError('LFW split is too small for the required 150 components.')
    write_status(folder,'running','Fitting training-only PCA and balanced RBF SVM.')
    pca=PCA(n_components=150,svd_solver='randomized',whiten=True,random_state=42)
    fit_start=time.perf_counter();xp=pca.fit_transform(X[train]);xt=pca.transform(X[test]);pca_s=time.perf_counter()-fit_start
    clf=SVC(kernel='rbf',class_weight='balanced');fit_start=time.perf_counter();clf.fit(xp,y[train]);svm_s=time.perf_counter()-fit_start
    pred=clf.predict(xt);names=lfw.target_names.tolist()
    report=classification_report(y[test],pred,labels=np.arange(len(names)),target_names=names,zero_division=0,output_dict=True)
    ev=pca.explained_variance_ratio_.cumsum();full_ev=ev
    if ev[-1]<.9:
        write_status(folder,'running','Calculating training-only full variance spectrum for the 90% threshold.')
        full=PCA(svd_solver='full').fit(X[train]);full_ev=full.explained_variance_ratio_.cumsum()
    k90=int(np.searchsorted(full_ev,.9)+1)
    def gallery(images,titles,name,cols=4):
        n=len(images);fig,axes=plt.subplots(max(1,(n+cols-1)//cols),cols,figsize=(10,max(2.8,2.5*((n+cols-1)//cols))))
        for ax in np.ravel(axes):ax.axis('off')
        for ax,img,title in zip(np.ravel(axes),images,titles):ax.imshow(img.reshape(h,w),cmap='gray');ax.set_title(title,fontsize=9)
        fig.tight_layout();fig.savefig(folder/name,dpi=140);plt.close(fig)
    faces=pca.components_.reshape(150,h,w)
    gallery(faces[:12],[f'Eigenface {i+1}' for i in range(12)],'eigenfaces.png')
    gallery(np.concatenate([faces[:3],faces[-3:]]),['PC1','PC2','PC3','PC148','PC149','PC150'],'first_last.png',3)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(np.arange(1,len(full_ev)+1),full_ev);ax.axhline(.9,color='#b34d32',linestyle='--');ax.axvline(k90,color='#116d60',linestyle=':');ax.set(xlabel='Components',ylabel='Cumulative explained variance',title=f'90% threshold: {k90} components');fig.tight_layout();fig.savefig(folder/'variance.png',dpi=140);plt.close(fig)
    wrong=np.flatnonzero(pred!=y[test]);chosen=wrong[:8]
    if len(chosen):gallery(X[test[chosen]],[f'True: {names[y[test[i]]].split()[-1]}\nPred: {names[pred[i]].split()[-1]}' for i in chosen],'failures.png')
    gallery(X[test[:12]],[f'True: {names[y[test[i]]].split()[-1]}\nPred: {names[pred[i]].split()[-1]}' for i in range(12)],'predictions.png')
    np.savez(folder/'evaluation.npz',train_indices=train,test_indices=test,y_true=y[test],y_pred=pred,variance=full_ev,components=pca.components_,mean=pca.mean_)
    joblib.dump({'pca':pca,'classifier':clf,'shape':[h,w],'classes':names},folder/'model.pkl')
    result={'samples':len(X),'features':X.shape[1],'shape':[h,w],'classes':names,'train_samples':len(train),'test_samples':len(test),'components':150,'variance_at_150':float(ev[-1]),'components_for_90_percent':k90,'pca_seconds':pca_s,'svm_seconds':svm_s,'total_seconds':time.perf_counter()-start,'report':report,'confusion_matrix':confusion_matrix(y[test],pred).tolist(),'failure_count':len(wrong),'failure_test_positions':chosen.tolist(),'seed':42}
    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    from reporting import pca_pdf
    pca_pdf(folder,result)
    write_status(folder,'complete','PCA, SVM, plots and digital PDF are ready.',result='result.json')

def yolo_job(folder,params):
    from dataset import validate_dataset
    import torch
    from ultralytics import YOLO
    audit=validate_dataset();device=0 if torch.cuda.is_available() else 'cpu'
    write_status(folder,'running',f'Training 100 epochs on {device}. Read job.log for progress.',device=str(device))
    model=YOLO(str(DATA/'models'/'yolo11n.pt'))
    def progress(trainer):
        write_status(folder,'running',f'Epoch {trainer.epoch+1} of 100',epoch=trainer.epoch+1,device=str(device))
        if (folder/'cancel.request').exists():trainer.stop=True
    model.add_callback('on_train_epoch_end',progress)
    model.train(data=str(DATA/'dataset'/'data.yaml'),epochs=100,imgsz=640,batch=4,seed=42,patience=0,workers=0,device=device,project=str(folder),name='training',exist_ok=True)
    if (folder/'cancel.request').exists():write_status(folder,'cancelled','Cancelled before held-out evaluation.');return
    best=folder/'training'/'weights'/'best.pt';trained=YOLO(str(best))
    # Model selection is validation-only; evaluate the held-out split once.
    test=trained.val(data=str(DATA/'dataset'/'data.yaml'),split='test',device=device,project=str(folder),name='heldout',plots=True)
    result={'device':str(device),'dataset':audit,'weights':str(best),'test_metrics':test.results_dict,'epochs':100,'imgsz':640,'seed':42}
    (folder/'result.json').write_text(json.dumps(result,indent=2,default=float),encoding='utf-8')
    write_status(folder,'complete','Training and held-out evaluation complete.',result='result.json')

def semantic_job(folder,params):
    # UI writes the input immediately after launch; don't race that write.
    deadline=time.monotonic()+60
    while not (folder/'input.ready').exists():
        if time.monotonic()>deadline:raise ValueError('Semantic input was not supplied within 60 seconds.')
        time.sleep(.1)
    import numpy as np
    import torch
    import cv2
    from PIL import Image
    from torchvision.models.segmentation import deeplabv3_resnet50,DeepLabV3_ResNet50_Weights
    from core import mask_scores
    write_status(folder,'running','Loading DeepLabV3 VOC weights; first use downloads model weights.')
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model=deeplabv3_resnet50(weights=weights).eval();device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    im=Image.open(folder/'input.png').convert('RGB');tensor=weights.transforms()(im).unsqueeze(0).to(device)
    start=time.perf_counter()
    with torch.inference_mode():
        logits=model(tensor)['out'];logits=torch.nn.functional.interpolate(logits,size=(im.height,im.width),mode='bilinear',align_corners=False);pred=logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    seconds=time.perf_counter()-start
    palette=np.random.default_rng(42).integers(40,245,(21,3),dtype=np.uint8);palette[0]=0
    rgb=palette[pred];overlay=(np.asarray(im)*.55+rgb*.45).astype(np.uint8)
    Image.fromarray(pred).save(folder/'labels.png');Image.fromarray(rgb).save(folder/'classes.png');Image.fromarray(overlay).save(folder/'overlay.png')
    cats=weights.meta['categories'];result={'model':'DeepLabV3-ResNet50','device':device,'inference_seconds':seconds,'legend':{str(int(c)):cats[c] for c in np.unique(pred)},'shape':list(pred.shape)}
    if (folder/'truth.png').exists():
        truth=np.asarray(Image.open(folder/'truth.png'))
        if truth.ndim!=2:raise ValueError('Ground truth must be an indexed or grayscale class-ID mask, not an RGB visualization.')
        if not set(np.unique(truth)).issubset(set(range(21))|{255}):raise ValueError('VOC mask must contain class IDs 0–20 or void 255.')
        result['scores']=mask_scores(pred,truth,21)
    (folder/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');write_status(folder,'complete','Semantic labels and overlay ready.',result='result.json')

if __name__=='__main__':
    kind,location=sys.argv[1:];folder=Path(location);folder.mkdir(parents=True,exist_ok=True)
    try:
        import platform
        from importlib.metadata import version,PackageNotFoundError
        versions={}
        for package in ['numpy','opencv-python','scikit-learn','torch','torchvision','ultralytics']:
            try:versions[package]=version(package)
            except PackageNotFoundError:pass
        (folder/'environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':versions,'kind':kind,'start_time':time.time()},indent=2),encoding='utf-8')
        params=json.loads((folder/'request.json').read_text())
        {'pca':pca_job,'yolo':yolo_job,'semantic':semantic_job}[kind](folder,params)
    except Exception as exc:
        traceback.print_exc();write_status(folder,'failed',str(exc))
