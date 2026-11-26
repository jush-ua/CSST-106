# Lab 06 Curated object detection experiment

Classes: 0=bottle, 1=cup, 2=phone. Source kind: curated_public. Annotation source: COCO instances_val2017; boxes imported, visually reviewed and corrected as needed. Raw images, annotations, videos and weights remain local. Source/capture groups were split and frozen before training.

COCO-pretrained weights have prior COCO exposure. This is a classroom fine-tuning exercise, not an independent benchmark.

YOLO11n: 100 total epochs, input size 640, CPU, batch 2, seed 42, no early stopping. Validation selected best.pt; the test split was evaluated once after training.

| Split | Images | Bottle instances | Cup instances | Phone instances |
|---|---:|---:|---:|---:|
| train | 180 | 129 | 130 | 81 |
| val | 39 | 21 | 31 | 23 |
| test | 38 | 26 | 28 | 18 |

Resume history (the batch above is the final batch):
- After epoch 68, resumed toward 100 total epochs with batch 2.

## Held-out metrics

- metrics/precision(B): 0.4081096067350487
- metrics/recall(B): 0.44485144485144484
- metrics/mAP50(B): 0.35573156242593723
- metrics/mAP50-95(B): 0.25345695984562017
- fitness: 0.25345695984562017

See results.csv, results.png and confusion_matrix.png and confusion_matrix_normalized.png for losses, validation trends and class errors. No arbitrary accuracy target is asserted. Small test sets and limited physical object variety restrict generalization.

## Separate demonstration images and optional recorded video

The PC has no webcam. The public still-image demonstration is not a live-webcam test. Recorded video can be processed separately; live webcam testing requires a connected camera. Recorded playback FPS is not inference throughput. Unlabelled demonstration images/video have no precision or recall ground truth. The custom head uses local IDs 0/1/2; pretrained COCO has 80 output IDs and calls phone "cell phone". Demo predictions are compared qualitatively, not as a second held-out evaluation.

Separate still-image outputs: 20 model/image comparisons.

Qualitative notes remain in the private local bundle.

## Detection versus classification

Classification assigns a label to an entire image. Detection locates and labels individual objects: a desk photo may contain a bottle, cup and phone with separate boxes, instead of one overall scene label.

See DEMO_ANALYSIS.md for the published qualitative review and metric interpretation when available.
