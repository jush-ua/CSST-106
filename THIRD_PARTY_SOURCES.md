# Data and library sources

The application's source is authored for this laboratory project. External libraries retain their own licenses; the dependency lockfile identifies installed versions.

- scikit-image public sample images: https://scikit-image.org/docs/stable/api/skimage.data.html . Included example outputs use Camera, Coins and Coffee, not the student's personal dataset.
- LFW data, retrieved by scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html . The PCA PDF and galleries are derived academic analysis; original image rights remain with their owners. Raw dataset files are not committed.
- PCA implementation reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html .
- OpenCV source and SURF capability: https://docs.opencv.org/4.12.0/d2/dca/group__xfeatures2d__nonfree.html . SURF setup pins both repositories to 4.12.0.
- Official YOLO11 checkpoint: https://docs.ultralytics.com/models/yolo11 . Ultralytics code/weights are distributed under their own AGPL-3.0/enterprise terms. Model weights are not committed here.
- DeepLabV3-ResNet50 weights: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html .
- VOC2012 original dataset: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/index.html . Small local validation subset retrieved from https://huggingface.co/datasets/nateraw/pascal-voc-2012 at revision 65505efaf0a3c4afe512dfc27c3abe2eac397bf9. Original PNG palette colours were mapped exactly to VOC class IDs. Row IDs and hashes are retained in the local subset manifest. Dataset files are excluded from Git.
- Assignment requirements: Jeremy C. Reyes's post and seven DOCX attachments in https://paios-classroom.com/classes/42 . Original instructions are referenced, not republished.
