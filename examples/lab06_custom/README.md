# Curated COCO Lab 06 experiment

The reviewed collection contains 257 unique COCO val2017 images, with 100 positive images per class (bottle/cup/phone) and 30 backgrounds. Mixed scenes count toward multiple class quotas. Another ten images are reserved for separate demonstration.

`curation.json` records source image IDs, URLs, per-image licences, original-image checksums, source groups, split assignments and actual annotation review/correction metadata. No downloaded photographs or individual box labels are redistributed here. The source phone box in image 380913 was corrected after full-resolution inspection.

The split contains 180 training, 39 validation and 38 test images. The reviewed split was frozen before training. Existing COCO source annotations were imported and reviewed; they are not claimed to have been drawn from scratch, and these are not personally photographed images.

The three-epoch CPU pilot completed at batch 4, with an estimate of about 4.1 hours for 100 training epochs plus startup/evaluation. Full training is underway. Final metrics and analysis will only be published after all 100 epochs and the held-out evaluation complete.

COCO-pretrained weights have prior COCO exposure; this is a classroom fine-tuning exercise rather than an independent generalization benchmark. A live-webcam demonstration has not been performed on this camera-less PC.
