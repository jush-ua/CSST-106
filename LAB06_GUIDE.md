# Lab 06 Curated public collection

This guide covers the Lab 06 dataset, annotation, training and inference workflow for CSST106. The experiment uses curated public COCO images, imported source boxes with visual review/correction, and local CPU training. Source images and annotation provenance are documented in examples/lab06_custom/curation.json. The repository URL has already been submitted to the class spreadsheet.

## Collection and annotation review

Run `.\.venv\Scripts\python.exe curate_coco.py` to download the official COCO 2017 annotation archive and only the selected val2017 JPEGs. The script resolves category IDs by source category names, remaps them to **0=bottle, 1=cup, 2=phone** (COCO calls the last category "cell phone"), imports every valid target box in each selected image and excludes target crowd annotations. Cup includes mugs; the class is not restricted to mugs.

The target is 100 positive images per class plus 30 backgrounds. Mixed scenes satisfy several quotas, so the collection can contain fewer than 330 unique images. Ten additional disjoint images are reserved for qualitative demonstration. Selection uses seed 42 and avoids oversized full image downloads. URLs, per-image licences, checksums, source boxes and selection settings are stored under ignored `data/coco_curated`. Do not remove these records.

In **Lab 06 > Curated dataset**, source boxes are initially unreviewed. Inspect labelled contact sheets and ambiguous originals at full resolution. Correct class/box errors in the editable annotation tool, then save and mark each inspected image reviewed. Background-only images need an explicitly reviewed empty label list. The command `curate_coco.py --review-ids ...` records review completion for the specified image IDs after inspection.

Source image IDs and perceptually near-duplicate clusters act as groups; each group represents a source image or related images. Keep a group in only one train/validation/test split. All three classes must appear in every split. Build and validate the grouped dataset before starting training. The legacy metadata key `session` stores source/capture groups for compatibility.

Personal import remains optional. If adding your own JPEG/PNG photos, use real capture-group names and annotate every visible bottle, cup and phone. Keep the bottle/cup/phone class mapping consistent across imports.

## Pilot and full training

```powershell
.\.venv\Scripts\python.exe run_lab.py 6 --pilot
.\.venv\Scripts\python.exe run_lab.py 6
```

The separate pilot runs three CPU epochs at 640 pixels, trying batch 4, then 2 or 1 on insufficient memory. Its final two epoch timings estimate the full run; initialization and evaluation add time. A pilot is not a completed Lab 06 model.

Full training starts fresh from pretrained YOLO11n and uses 100 total epochs, image size 640, seed 42, no early stopping and the successful matching pilot batch. Training freezes images, labels, class mapping, provenance and split metadata. Keep the PC awake and D: available throughout training.

Use **Pause after saved epoch**, then **Resume original run**, or:

```powershell
.\.venv\Scripts\python.exe run_lab.py 6 --resume runs\job_yolo_YOUR_RUN
```

If a full run fails for insufficient memory, add `--batch 2` (or `--batch 1`) to the resume command. This changes memory use while retaining the original total epoch target; each restart is recorded in `resume_history.json`.

Resume retains optimizer state and continues toward 100 total epochs. Two committed checkpoint generations are retained with an atomic progress pointer; a force stop can lose the unfinished epoch. Dataset changes block old-run resume. Unfreeze only to make corrections and start a newly compatible experiment.

Validation selects best.pt. The held-out test split is evaluated once after all 100 epochs. An interrupted held-out evaluation is flagged rather than silently repeated. Inspect precision, recall, mAP50, mAP50-95, loss curves and confusion matrices; interpret them alongside the error examples.

**Interpretation limit:** COCO-pretrained weights have prior COCO exposure, including historical model selection on COCO validation. These results demonstrate classroom fine-tuning on a curated subset, not an independent benchmark or proof of generalization to an entirely unseen dataset. Source annotations can contain ambiguity, very small objects and stylized depictions.

## Separate still-image demonstration and optional video

Use the ten reserved public images, never training/validation/test images. Compare pretrained and custom predictions qualitatively, record successes, missed objects and false alarms, and keep original COCO source records for attribution.

```powershell
.\.venv\Scripts\python.exe lab06.py demo --job runs\job_yolo_YOUR_RUN --images C:\path\new1.jpg C:\path\new2.jpg
# Optional separate recorded clip:
.\.venv\Scripts\python.exe lab06.py demo --job runs\job_yolo_YOUR_RUN --video C:\captures\demo.mp4 --images C:\path\new1.jpg
```

Only evaluated 100-epoch custom models appear in the app's model selector. Demonstration still images are rejected if their decoded pixels duplicate any exported dataset image. Demo predictions are not a second held-out test evaluation. Add actual qualitative observations to the job's `demo/review.txt` and regenerate the report.

The PC has no webcam. A separate recorded clip can be processed through the video inference feature. Live webcam testing requires a connected camera and remains unperformed. Recorded playback FPS is not inference throughput.

## Deliverables and publication

```powershell
.\.venv\Scripts\python.exe lab06.py report --job runs\job_yolo_YOUR_RUN
.\.venv\Scripts\python.exe lab06.py bundle --job runs\job_yolo_YOUR_RUN
.\.venv\Scripts\python.exe lab06.py publish-summary --job runs\job_yolo_YOUR_RUN
```

The private ZIP contains best.pt, metrics, plots, analysis and available demonstration outputs. Images, individual labels, recordings and weights stay local. Public output contains source/licence/image-ID records, split assignments, annotation-review/correction metadata, numerical metrics, analysis and training/confusion charts. It does not redistribute downloaded photographs or weights. Review notes remain private; publish a separate non-identifying qualitative analysis when appropriate.
