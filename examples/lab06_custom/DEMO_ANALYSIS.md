# Reserved-image comparison and error analysis

The ten COCO val2017 source IDs below were excluded from training, validation and test before training. All 20 annotated outputs (pretrained/custom) were visually reviewed. Inference used confidence 0.25 and only bottle/cup/phone categories. These are qualitative examples, not an additional test evaluation. Photographs and annotated outputs remain in the private local bundle; attribution and source links are in curation.json.

| COCO image ID | Pretrained boxes | Custom boxes | Observations |
|---|---:|---:|---|
| 51598 | 0 | 0 | Both missed the small bathroom bottle. |
| 458663 | 0 | 1 | Custom produced a low-confidence cup box over part of the distant cup group; individual cups were not separated. |
| 338304 | 0 | 0 | Both missed the small phone in a crowded outdoor scene. |
| 472678 | 1 | 1 | Both localized the prominent desk bottle (0.72/0.75 confidence); neither found the desk phone. |
| 325483 | 3 | 1 | Both detected the foreground bottle. Custom missed the distant bottle/cup; pretrained produced additional boxes there. |
| 336628 | 0 | 0 | Both missed the very small, partially obscured phone. |
| 192670 | 0 | 0 | Both missed the distant bottle near the background spectators. |
| 436617 | 1 | 4 | Custom produced cup boxes on foreground drinking vessels, including two overlapping boxes on one vessel. The source has one cup annotation; extra glassware boxes need caution because cup/glass boundaries and source completeness are ambiguous. |
| 325838 | 0 | 0 | Both missed the small phone beside the laptop. |
| 395701 | 0 | 0 | Both missed the two small bottles at the right of the room. |

More boxes do not establish better accuracy. Neither model produced a phone detection on these ten images at this confidence. The results show sensitivity to object size, clutter, occlusion and category ambiguity. They do not establish that fine-tuning improved the pretrained model overall.

## Training and held-out results

The frozen collection has 180 training, 39 validation and 38 test images. Training completed 100 epochs at 640 pixels on CPU, seed 42, without early stopping. Epochs 1–68 used batch 4. CPU allocation failed during epoch 69; native resume retained the saved optimizer and continued epochs 69–100 at batch 2. The CSV contains exactly epochs 1–100. Its time column restarts at resume and should not be interpreted as a continuous wall clock.

Validation mAP50–95 peaked at epoch 63 (0.32169), which selected the saved best model. The final epoch validation mAP50–95 was 0.29154. Training box/class loss fell from 1.20961/2.31918 to 0.70407/0.61503, while validation box loss rose from 0.95577 to 1.40719. This gap and the earlier validation peak suggest overfitting; lower training loss does not establish improved generalization.

The selected model was evaluated once on the 38-image held-out split: precision 0.40811, recall 0.44485, mAP50 0.35573 and mAP50–95 0.25346. These library detection metrics are not classification accuracy. The exported confusion chart uses the installed validator's low confidence threshold 0.001 and matching IoU 0.5; its many background false positives include low-confidence predictions. It must not be treated as a confusion chart for the demonstration's 0.25 threshold or as the scalar precision operating point.

COCO-pretrained weights have prior COCO exposure. The small curated COCO validation subset is a classroom fine-tuning exercise, not an independent benchmark. Possible future improvements include more diverse independent images, clearer cup/glass annotation rules, additional reviewed small-object examples, and validation-based confidence tuning. No such changes were made after examining the held-out results.

## Hardware-dependent requirement

No live webcam demonstration was possible because this PC has no webcam. The reserved still images do not fulfill that live step. The ten reserved still images demonstrate the new-image inference step. Recorded video is an optional additional inference input.
