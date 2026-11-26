# CSST106 Vision laboratory

A local Windows workbench for the seven laboratory attachments in Jeremy C. Reyes's class announcement. This project is separate from the 63 proctored activities. No printed or handwritten report is included.

## Start

Open PowerShell in this folder and run:

```powershell
.\setup.ps1
.\start.ps1
```

Open **http://127.0.0.1:8501**. Python 3.11 is required. Installation and first use of LFW/YOLO/DeepLab need network access. The environment is local to `.venv`; the application binds only to loopback. All dataset caches and model caches are kept under `data/` where configured. Do not move or delete the project while its jobs are running.

After setup, you can also double-click `Launch.cmd`. Labs 05 and 07 offer a completed reference experiment from `examples/`, so the measured PCA/PDF and semantic results can be reviewed without rerunning model downloads. These are public-sample results; Lab 06 uses a separately documented curated training collection.

## Modules

| Lab | Required implementation | Location |
|---|---|---|
| 01 | BGR pixels, dimensions, grayscale, editing, crop, resize, 90° rotation | `core.manipulate` |
| 02 | Blur 3×3, float Sobel x/y and magnitude, Canny 100/200 | `core.edges` |
| 03 | Histograms, linear stretching, global equalization, CLAHE, YUV luminance mode | `core.contrast` |
| 04 | SURF matching, Hessian sweep, U-SURF, SIFT/ORB comparison | `core.features`, `surf_worker.py` |
| 05 | LFW, training-only 150-component PCA, whitening, balanced RBF SVM, galleries, variance, PDF | `worker.pca_job`, `reporting.py` |
| 06 | YOLO11 image/video/webcam, curated or personal annotation, grouped split, 100-epoch training and held-out evaluation | `dataset.py`, `worker.yolo_job` |
| 07 | Otsu/adaptive, region growing, watershed, contours, DeepLab semantic labels and optional IoU | `core.segmentation`, `worker.semantic_job` |

Use **Save experiment** to retain images, settings, metrics and dependency versions. Lab 02 saves signed gradients and raw magnitude as NPY arrays. Background jobs remain active when navigating. Their logs and status are on disk in `runs/job_*`; use the job's Cancel button to terminate it. Restarting the app does not silently retrain models. Video processing is synchronous and should use short clips.

Use `python run_lab.py --help` for commented standalone lab entry points. Individual algorithms are importable for inspection. The required PCA script is `worker.py`, not a disconnected notebook.

## Completed experiments and analysis

Labs 01-03 and 07 include measured digital analyses under `examples/lab01/ANALYSIS.md`, `examples/lab02/ANALYSIS.md`, `examples/lab03/ANALYSIS.md` and `examples/lab07/ANALYSIS.md`. Open **Completed reference analysis and figures** on each lab page to read the analysis and download a comparison figure. Original arrays, settings, dependency versions and image hashes remain in the case manifests.

Lab 01 uses the actual colour Astronaut sample, including a visible pixel edit and all image operations. Lab 02 exports exactly original/Sobel/Canny side by side. Lab 03 covers low contrast, uneven illumination with noise, and colour luminance; each case exports four images, histograms and all histogram bins as CSV. Lab 07 compares five classical methods across touching objects, uneven illumination, texture and a natural VOC scene, including seed/marker sensitivity, compatible foreground scores and timings. DeepLab runs on that same natural scene with separate class-based scoring. Controlled fixtures are explicitly labelled; none are personal YOLO data.

Reproduce all four labs after fetching the VOC subset and caching the DeepLab weights:

```powershell
.\.venv\Scripts\python.exe complete_labs.py
# Individual labs or an instructor-supplied edge image:
.\.venv\Scripts\python.exe complete_labs.py --labs 1 3
.\.venv\Scripts\python.exe complete_labs.py --labs 2 --edge-image C:\path\input.jpg
```

Lab 02 now bundles a real `examples/lab02/input.jpg`: the public scikit-image Camera sample encoded as JPEG quality 95. `input_source.json` records its source and checksum and explicitly marks it as a replacement, not instructor-provided. The UI, `run_lab.py 2` and evidence generator use this actual decoded JPEG by default; explicit uploads/files or sample selections override it. All figures, gradients and metrics were regenerated after JPEG compression. Jeremy's original file remains unavailable; use `--edge-image` if it is later supplied.

## Curated public dataset and custom YOLO

Lab 06 uses a curated **COCO 2017 validation collection**, not personal photographs. Classes are 0 bottle, 1 cup, 2 phone (source category "cell phone"). Existing COCO annotations are imported, visually reviewed and corrected where necessary. See [LAB06_GUIDE.md](LAB06_GUIDE.md) for the complete workflow.

Run `.\.venv\Scripts\python.exe curate_coco.py` to download only the selected JPEGs and the official annotation archive. Imported images start unreviewed; the annotation tool and private contact sheets support inspection and corrections. The selected collection has 257 unique images: 100 positive images per class with mixed scenes counting toward multiple quotas, plus 30 backgrounds. Ten additional disjoint images are reserved for qualitative demonstration. Source URLs, licence records, checksums and reviewed split membership are published in `examples/lab06_custom/curation.json`; downloaded pixels, individual labels and weights remain local.

Source image groups and near-duplicate clusters are kept in one split. The actual split is 180 training, 39 validation and 38 test images, with all classes in every split. The dataset is frozen and fingerprinted before training. COCO-pretrained weights have prior COCO exposure, so results describe a classroom fine-tuning exercise rather than an independent benchmark.

Run `run_lab.py 6 --pilot` for a separate three-epoch CPU pilot, then `run_lab.py 6` for fresh 100-epoch YOLO11n training at 640 pixels, seed 42 and no early stopping. Batch starts at 4 with memory fallback in the pilot. Pause/resume retains optimizer state and continues toward 100 total epochs. Validation chooses best.pt; the held-out test split is evaluated once after all epochs. A pilot is never listed as a completed custom model.

`lab06.py demo --job runs/job_yolo_ID --images ...` compares pretrained and custom predictions on the reserved still images, rejects exact dataset-image reuse, and records outputs for qualitative error analysis. A separate phone-recorded video remains optional and cannot be represented as live-webcam footage; acceptance of a substitute for Jeremy's live step remains pending. Private bundles include weights and available demo outputs. `publish-summary` copies only provenance, metrics, analysis and training/confusion charts for GitHub.

Personal photo import remains available as an alternative, using real capture-group names and the current cup class. Do not reinterpret old mug-only records without checking their class mapping.

## SURF

Stock OpenCV wheels usually do not enable SURF. Install the official Visual Studio 2022 C++ Build Tools and Windows SDK, then run:

```powershell
.\setup-surf.ps1
```

This builds matching OpenCV and opencv_contrib 4.12.0 with `OPENCV_ENABLE_NONFREE=ON` in `.surf-venv`. Only the SURF worker imports that build; it does not replace the main cv2. Use `-BuildRoot D:\CSST106-build\opencv-src` to place sources and compilation on another drive. The script verifies detector construction. A failed build leaves SURF explicitly incomplete rather than presenting ORB as SURF. Source: https://docs.opencv.org/4.12.0/d2/dca/group__xfeatures2d__nonfree.html

After building, run `.\.venv\Scripts\python.exe verify_surf.py` to reproduce the threshold sweep and SURF/U-SURF/SIFT/ORB comparisons on rotation and translation pairs. The evidence is written under `examples/lab04`, including measured tables and interpretation in `ANALYSIS.md`. All comparison algorithms run in the same isolated OpenCV build. That worker exchanges PNG images; its reduced codec support does not change the main application's codecs.

The standalone command also runs the full nonfree comparison and saves all detector images and sweep records:

```powershell
.\.venv\Scripts\python.exe run_lab.py 4 --image a.jpg --second b.jpg --hessian 400 --ratio .75
# Without --second, use a 90-degree rotated copy; add --upright to select U-SURF images.
.\.venv\Scripts\python.exe run_lab.py 4 --sample Camera --upright
```

The UI and command share the same worker bridge. The command exits with an error if SURF is unavailable; it does not claim a complete Lab 04 run using only SIFT/ORB.

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

The edge attachment names input.jpg but no standalone image was attached. A disclosed replacement JPEG is now bundled and is the default; the original can still be supplied as an override. Original DOCX documents remain in the user's Downloads folder; they are not automatically republished.

## Video and webcam verification

`examples/media_verification/verification.json` records a real YOLO video run: all 12 frames decoded successfully at the source 12 FPS. The fixture and annotated output are included. Reproduce with `.\.venv\Scripts\python.exe verify_media.py`; add `--webcam` only when a camera is connected. This computer has no webcam, so live camera capture has not been verified. No camera images were saved.

## Verification and handoff

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe run_lab.py 2 --sample Camera
```

Read `IMPLEMENTATION_STATUS.md` for actual verification and remaining requirements. Tests use temporary synthetic grids, never fake personal training results. A failed download/build, missing reviewed dataset or unfinished training must remain visible as incomplete.

The selected destination is https://github.com/jush-ua/CSST-106. Source publication status is recorded in `IMPLEMENTATION_STATUS.md`. Commit source, lockfile, README, small curated outputs and the PCA PDF; use Git LFS or release assets for selected large weights. Do not commit `.venv`, raw personal datasets or all training runs. Jeremy's comment requests a GitHub URL in the linked class spreadsheet. **The user confirmed that the repository URL has already been submitted.** No additional spreadsheet entry is needed.

Lab 06 still-image/video comparison, reports and the private weights/output bundle are produced with `lab06.py`; see the curation guide for exact commands. Raw inputs, annotation sheets, videos and weights stay local. `publish-summary` explicitly copies only numerical metrics, analysis and training/confusion charts into `examples/lab06_custom` for source publication. Recorded phone video is disclosed as a webcam substitute, with instructor acceptance still pending.
