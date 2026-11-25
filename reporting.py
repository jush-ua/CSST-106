"""PCA digital analysis generated from the actual experiment, never mock scores."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

def pca_pdf(folder,r):
    folder=Path(folder);styles=getSampleStyleSheet();story=[]
    note_file=folder/'visual_analysis.json'
    notes=json.loads(note_file.read_text(encoding='utf-8')) if note_file.exists() else {}
    def para(text,style='BodyText'):story.append(Paragraph(text,styles[style]));story.append(Spacer(1,8))
    def image(name,width=470):
        p=folder/name
        if p.exists():
            w,h=ImageReader(str(p)).getSize();story.append(Image(str(p),width=width,height=width*h/w));story.append(Spacer(1,10))
    para('PCA Eigenfaces Experiment and Analysis','Title')
    para(f"This experiment recognizes LFW identities using a training-only 150-component whitened PCA projection and a balanced RBF SVM. The held-out accuracy is {r['report']['accuracy']:.3f}; {r['failure_count']} test images were misclassified. All figures and metrics below come from this run.")
    para('Method and measured results','Heading1')
    para(f"LFW: {r['samples']} faces, {len(r['classes'])} identities, {r['features']} flattened features. Images are {r['shape'][0]} by {r['shape'][1]}. The 80/20 split has seed 42, {r['train_samples']} training examples and {r['test_samples']} test examples. PCA uses randomized SVD, whitening, 150 components and seed 42. SVM uses an RBF kernel and balanced class weights. PCA and SVM fitting took {r['pca_seconds']:.2f} and {r['svm_seconds']:.2f} seconds respectively.")
    rows=[['Identity','Precision','Recall','F1','Support']]
    for name in r['classes']:
        q=r['report'][name];rows.append([name,f"{q['precision']:.3f}",f"{q['recall']:.3f}",f"{q['f1-score']:.3f}",str(int(q['support']))])
    t=Table(rows,colWidths=[180,70,70,70,65],repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#234B45')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F5F4')]),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t)
    para('Representative held out predictions','Heading1');image('predictions.png',360)
    story.append(PageBreak());para('Dimensionality and variance','Heading1')
    para(f"The required 150-component model retains {r['variance_at_150']*100:.2f}% of training variance. The first {r['components_for_90_percent']} components reach 90%. Where 150 is insufficient, a separate full-spectrum training-only PCA determines this threshold; the classifier remains at 150 components.")
    image('variance.png');para('Visual interpretation of eigenfaces','Heading1');image('first_last.png')
    para(escape(notes.get('eigenfaces', 'The first components describe the greatest training variation; inspect their broad illumination and shape patterns, then compare with finer residual patterns in the last components. Eigenvector sign is arbitrary. Record image-specific interpretations in the application.')))
    story.append(PageBreak());para('Whitening and the SVM','Heading1')
    para('PCA first subtracts the training feature mean and rotates into orthogonal principal directions. Whitening additionally scales each projected coordinate by the inverse square root of its estimated variance, producing unit sample variance for nonzero components. This prevents high-variance principal directions from dominating Euclidean distances used by the RBF kernel. Whitening can help the SVM, but it can also amplify weak noisy directions and is not a guarantee of higher accuracy. No test labels are used to fit this transformation.')
    para('Failure analysis','Heading1');image('failures.png')
    if r['failure_count']:
        para(escape(notes.get('failure_analysis', 'These failures were selected programmatically from unequal true and predicted labels. Compare each displayed error with the eigenfaces and record image-specific hypotheses involving pose, lighting, expression or occlusion. These are possible contributors, not proven causes.')))
    else:para('No held-out errors occurred in this run; no failed examples have been manufactured.')
    para('Follow-up questions','Heading1')
    para('Dimensionality reduction lowers storage and computation and can remove noise. PC1 captures the greatest remaining variance; PC2 is orthogonal and captures the next greatest. Reducing 1,000 features to 100 components discards variation outside the retained subspace, which may include identity details as well as noise. Face recognition projects centered faces into the learned training subspace and classifies their coefficients. Centering ensures PCA describes variation around the mean rather than absolute image brightness.')
    para('References','Heading1');para('Laboratory attachment 05; scikit-learn PCA documentation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html. Dataset: Labeled Faces in the Wild, via fetch_lfw_people(min_faces_per_person=70, resize=0.4).')
    SimpleDocTemplate(str(folder/'PCA_analysis.pdf'),pagesize=(612,792),rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42,title='PCA Eigenfaces Experiment and Analysis').build(story)
