# CSST106 Vision laboratory

A local Windows workbench for the seven laboratory attachments in Jeremy C. Reyes's class announcement. This project is separate from the 63 proctored activities. No printed or handwritten report is included.

## Start

Open PowerShell in this folder and run:

```powershell
.\setup.ps1
.\start.ps1
```

Open **http://127.0.0.1:8501**. Python 3.11 is required. Installation and first use of LFW/YOLO/DeepLab need network access. The environment is local to `.venv`; the application binds only to loopback. All dataset caches and model caches are kept under `data/` where configured. Do not move or delete the project while its jobs are running.

After setup, you can also double-click `Launch.cmd`. Labs 05 and 07 offer a completed reference experiment from `examples/`, so the measured PCA/PDF and semantic results can be reviewed without rerunning model downloads. These are public-sample results, not custom personal training outcomes.

## Modules

| Lab | Required implementation | Location |
|---|---|---|
| 01 | BGR pixels, dimensions, grayscale, editing, crop, resize, 90° rotation | `core.manipulate` |
| 02 | Blur 3×3, float Sobel x/y and magnitude, Canny 100/200 | `core.edges` |
| 03 | Histograms, linear stretching, global equalization, CLAHE, YUV luminance mode | `core.contrast` |
| 04 | SURF matching, Hessian sweep, U-SURF, SIFT/ORB comparison | `core.features`, `surf_worker.py` |
| 05 | LFW, training-only 150-component PCA, whitening, balanced RBF SVM, galleries, variance, PDF | `worker.pca_job`, `reporting.py` |
| 06 | YOLO11 image/video/webcam, personal annotation, grouped split, 100-epoch training and held-out evaluation | `dataset.py`, `worker.yolo_job` |
| 07 | Otsu/adaptive, region growing, watershed, contours, DeepLab semantic labels and optional IoU | `core.segmentation`, `worker.semantic_job` |

Use **Save experiment** to retain images, settings, metrics and dependency versions. Lab 02 saves signed gradients and raw magnitude as NPY arrays. Background jobs remain active when navigating. Their logs and status are on disk in `runs/job_*`; use the job's Cancel button to terminate it. Restarting the app does not silently retrain models. Video processing is synchronous and should use short clips.

Use `python run_lab.py --help` for commented standalone lab entry points. Individual algorithms are importable for inspection. The required PCA script is `worker.py`, not a disconnected notebook.

## Personal dataset and custom YOLO

Lab 06 → Personal dataset imports **your own** bottle/mug/phone images. Demo samples are not personal training data. Target 100 distinct images per class plus 30 backgrounds. Collect at least three separate sessions, preferably ten or more, with all classes in multiple sessions. Avoid near-identical consecutive frames across sessions.

Select each photo, click opposite corners, choose the class and add a box. Repeat for every visible target. The editable table permits corrections or deletion. Save and mark reviewed; reviewed empty boxes mean background-only. Build the dataset, inspect the session split, and validate before training. Entire sessions are kept apart; exact 70/15/15 image proportions are approximate when sessions differ in size. The split builder rejects any split missing a class.

Class IDs: 0 bottle, 1 mug, 2 phone. Labels are normalized `class_id x_center y_center width height`. The generated YAML includes train/val/test, with explicit class names. Training uses `yolo11n.pt`, 100 epochs, 640 pixels, batch 4, seed 42, and `patience=0`. Validation selects best.pt. The test split is evaluated once after training. CPU is supported but can be slow; a CUDA-capable PyTorch installation is used automatically when available. No accuracy target is invented.

Raw photos, annotations, large weights and runs are ignored by Git. Preserve them locally and share only the materials you intend the instructor to access. The annotation tool does not upload photos anywhere. The project can run entirely locally after datasets and weights are downloaded.

## SURF

Stock OpenCV wheels usually do not enable SURF. Install the official Visual Studio 2022 C++ Build Tools and Windows SDK, then run:

```powershell
.\setup-surf.ps1
```

This builds matching OpenCV and opencv_contrib 4.12.0 with `OPENCV_ENABLE_NONFREE=ON` in `.surf-venv`. Only the SURF worker imports that build; it does not replace the main cv2. Use `-BuildRoot D:\CSST106-build\opencv-src` to place sources and compilation on another drive. The script verifies detector construction. A failed build leaves SURF explicitly incomplete rather than presenting ORB as SURF. Source: https://docs.opencv.org/4.12.0/d2/dca/group__xfeatures2d__nonfree.html

After building, run `.\.venv\Scripts\python.exe verify_surf.py` to reproduce the threshold sweep and SURF/U-SURF/SIFT/ORB comparisons on rotation and translation pairs. The evidence is written under `examples/lab04`, including measured tables and interpretation in `ANALYSIS.md`. All comparison algorithms run in the same isolated OpenCV build. That worker exchanges PNG images; its reduced codec support does not change the main application's codecs.

## PCA analysis

Use Lab 05's Run button. Outputs include the actual classification report, sample and feature counts, first 12 eigenfaces, first/last component comparison, variance curve, predictions, programmatic errors, saved model, and `PCA_analysis.pdf`. The classifier retains the required 150 components even if a separate full training spectrum is needed for the 90% question. Fit uses the source's unstratified 80/20 split with random_state 42. No test images or labels fit the PCA or SVM.

The PDF includes measured results and conceptual analysis. Record your image-specific interpretation in the app; uncertainty about causes of recognition errors must remain a hypothesis.

## Segmentation ground truth

Binary truth must be a grayscale 0/1 or 0/255 image aligned exactly to the input. Semantic truth must contain VOC IDs 0–20 and void 255. Mismatched dimensions are rejected; labels are never silently interpolated. Semantic scoring excludes void pixels and omits absent classes from mean IoU. Watershed region IDs are not semantic classes.

The supplied local VOC subset was retrieved using `python fetch_voc_subset.py` from the pinned nateraw/pascal-voc-2012 mirror revision. Five validation rows and their hashes/provenance are recorded in `data/voc_subset/manifest.json`. Lossless RGB masks are converted using the exact VOC bit palette; unknown colours are rejected. Select a `VOC:` sample in Lab 07 to attach its ground truth automatically. These data are excluded from Git; run the script again after cloning. Alternatively, `python prepare_voc.py --root PATH_TO_VOC2012` selects five pairs from an official extracted dataset. Natural scikit-image samples demonstrate inference without ground-truth metrics.

## Source instructions and numbering

The attachment order is authoritative for navigation. Edge detection internally says Exercise 3; histogram also says 3; segmentation says 6. The YOLO filename says YOLOV8 but its body requires YOLOv11; official weights are named `yolo11n.pt`.

1. https://paios-classroom.com/announcement-attachments/1371/file
2. https://paios-classroom.com/announcement-attachments/1372/file
3. https://paios-classroom.com/announcement-attachments/1373/file
4. https://paios-classroom.com/announcement-attachments/1374/file
5. https://paios-classroom.com/announcement-attachments/1375/file
6. https://paios-classroom.com/announcement-attachments/1376/file
7. https://paios-classroom.com/announcement-attachments/1377/file

The edge attachment names input.jpg but no standalone image was attached. The UI accepts that image when supplied and labels its alternate public sample. Original DOCX documents remain in the user's Downloads folder; they are not automatically republished.

## Video and webcam verification

`examples/media_verification/verification.json` records a real YOLO video run: all 12 frames decoded successfully at the source 12 FPS. The fixture and annotated output are included. Reproduce with `.\.venv\Scripts\python.exe verify_media.py`; add `--webcam` only when a camera is connected. This computer has no webcam, so live camera capture has not been verified. No camera images were saved.

## Verification and handoff

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe run_lab.py 2 --sample Camera
```

Read `IMPLEMENTATION_STATUS.md` for actual verification and remaining requirements. Tests use temporary synthetic grids, never fake personal training results. A failed download/build or missing personal dataset must remain visible as incomplete.

The selected destination is https://github.com/jush-ua/CSST-106. Source publication status is recorded in `IMPLEMENTATION_STATUS.md`. Commit source, lockfile, README, small curated outputs and the PCA PDF; use Git LFS or release assets for selected large weights. Do not commit `.venv`, raw personal datasets or all training runs. Jeremy's comment requests a GitHub URL in the linked class spreadsheet; nothing has been posted there by this app.
