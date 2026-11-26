# Lab 06 Phone capture and local training

The spreadsheet repository URL has already been submitted, as confirmed by you. This workflow uses your phone for capture and this PC for annotation, training and recorded-video inference. Jeremy explicitly asks for a live webcam demonstration: acceptance of a disclosed recorded-phone-video substitute remains pending. Do not present recorded playback as a live webcam result.

## Capture your own dataset

Create ten folders named `session-01` through `session-10`. Each is a genuinely different capture setup: change the background, lighting, distance, angle and arrangement. Keep a burst or continuous setup in one session. Do not distribute near-identical frames across sessions.

For each session, take 10 bottle photos, 10 mug photos, 10 phone photos and 3 background-only photos. That is 33 images per session and 330 images total. This is our practical target, not an image count specified by Jeremy. Include front/side/angled views, different scales, partial occlusion and some mixed-object scenes. Use several physical examples if available; if not, acknowledge the limited object variety when interpreting results.

To photograph the recording phone, use a mirror or briefly borrow another camera. Label the visible phone, not the whole mirror. Transfer original JPEG/PNG images by USB; select JPEG capture/export if the phone normally produces HEIC. Avoid messaging-app recompression, screenshots and downloaded substitutes. Store originals outside the repository or under ignored `data/`.

Separately record a 15-30-second MP4 clip in a new arrangement. Include the target objects, changes in distance and a brief occlusion. Keep it out of training/validation/test image extraction. Record several new still photographs for the final demonstration as well.

## Import and annotate

1. Start the application and open **Lab 06 > Personal dataset**.
2. Enter the correct session name, select that session's JPEG/PNG files and import them. Repeat for all ten sessions.
3. For each image, set the opposite box corners, choose the class and add a tight box around every visible target. IDs are **0 bottle, 1 mug, 2 phone**. In mixed scenes annotate all target instances.
4. Correct boxes in the editable table and choose **Save boxes and mark reviewed**. For a true background-only image, save a reviewed empty list. An unreviewed image is not a negative example.
5. Generate the private labelled contact sheets and inspect coverage, class names and box boundaries. The session progress table counts reviewed photos and class presence.
6. Build the session-separated dataset and validate it. Ten equally sized sessions yield seven training sessions, two validation sessions and one test session: approximately 231/66/33 images. Actual assignment is deterministic with seed 42, not filename order.

YOLO labels contain `class_id x_center y_center width height`, with coordinates normalized to image size. The application creates image/label folders and `data.yaml`; do not hand-move similar frames between splits to improve scores.

## Pilot then train

Open **Train and evaluate** and run the **3-epoch pilot** first. It freezes the exported dataset and times actual CPU epochs at 640 pixels. It tries batch 4, then 2, then 1 on insufficient-memory errors. Read its estimated 100-epoch time; initialization and final evaluation add time. Pilot weights are never a completed custom model.

Start **full training** afterwards. It starts fresh from pretrained `yolo11n.pt`, uses the successful matching pilot batch, and runs 100 total epochs at 640 pixels, seed 42, CPU and no early stopping. Keep the PC awake and D: available during the job; page navigation does not cancel it. No Windows power settings are changed automatically.

```powershell
.\.venv\Scripts\python.exe run_lab.py 6 --pilot
.\.venv\Scripts\python.exe run_lab.py 6
```

Use **Pause after saved epoch** for a planned interruption, then **Resume original run**. The optimizer-bearing resume checkpoint is retained separately from inference weights; resume continues toward 100 total epochs. Force cancellation retains the most recent fully copied checkpoint and can lose the current unfinished epoch. A power failure during saving can require inspection of the last intact checkpoint.

```powershell
.\.venv\Scripts\python.exe run_lab.py 6 --resume runs\job_yolo_YOUR_RUN
```

Do not change the frozen dataset between pause and resume. Unfreezing permits corrections and rebuilding, but changed images, labels, YAML or split metadata make an old run incompatible. Completed 100-epoch training selects `best.pt` using validation data, then evaluates the held-out test split once. An interrupted held-out evaluation is flagged for explicit recovery rather than silently repeated. Review precision, recall, mAP50, mAP50-95, loss curves and confusion matrices; no invented accuracy target is required.

## Demonstrate the trained model and prepare delivery

Only evaluated 100-epoch models appear in the application model selector. You can process your clip there, or use the evidence command to compare pretrained and custom weights and verify saved video decoding:

```powershell
.\.venv\Scripts\python.exe lab06.py demo --job runs\job_yolo_YOUR_RUN --video C:\captures\demo.mp4 --images C:\captures\new1.jpg C:\captures\new2.jpg
```

Inspect the annotated outputs for actual successful detections, missed objects and false alarms. Record image-specific observations in the job's `demo/review.txt`, then regenerate the report. Do not report precision/recall for unlabelled demonstration video; those metrics come from the held-out labelled set. Source-video playback FPS and measured processing throughput are reported separately.

```powershell
.\.venv\Scripts\python.exe lab06.py report --job runs\job_yolo_YOUR_RUN
.\.venv\Scripts\python.exe lab06.py bundle --job runs\job_yolo_YOUR_RUN
# Explicitly copy only non-personal numerical results and training charts for Git:
.\.venv\Scripts\python.exe lab06.py publish-summary --job runs\job_yolo_YOUR_RUN
```

The private ZIP includes best.pt, metrics, training curves, the digital analysis and annotated demonstrations when available. Raw capture photos, labels, recordings, contact sheets and weights remain local and ignored by Git. The public summary omits weights, captured images/video and private qualitative notes. A training-complete model still needs its separate new-input demonstration and review; the report keeps these pending until the evidence exists.
