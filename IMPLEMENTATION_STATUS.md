# Implementation and verification status

Updated 26 November 2026. This CSST106 project implements seven computer vision laboratory experiments in a local Streamlit app. Digital experiments and analyses are complete. Printed reports and live webcam testing are outside the project scope.

## Laboratory results

| Lab | Implementation and saved evidence |
|---|---|
| 01 - Image manipulation | Colour Astronaut input, dimensions and BGR values, pixel editing, grayscale, crop, resize and 90-degree rotation; six-panel comparison and analysis. |
| 02 - Edge detection | Student-selected Camera-derived input.jpg, Gaussian blur, signed Sobel derivatives and magnitude, Canny edges, three-panel comparison and analysis. Source and checksum are recorded. |
| 03 - Histogram and contrast | Low-contrast, uneven-illumination/noise and colour-YUV experiments; contrast stretching, global equalization, CLAHE, histogram figures/CSVs and analysis. |
| 04 - Feature matching | Nonfree SURF/U-SURF threshold sweeps, SIFT/ORB comparisons, keypoint and match figures on rotation and translation pairs. UI and standalone command use the same worker. |
| 05 - PCA eigenfaces | LFW/PCA/SVM experiment, classification report, variance curve, eigenfaces, prediction/error galleries and digital PDF covering analysis and follow-up questions. |
| 06 - YOLO detection | Reviewed/corrected public COCO annotations, 100 training epochs at 640 pixels, one held-out evaluation, ten reserved-image comparisons, metrics, charts and error analysis. Trained weights and image outputs are in the local deliverables ZIP. |
| 07 - Segmentation | Otsu/adaptive thresholding, region growing, watershed, contour-based regions and DeepLab semantic segmentation; controlled, textured and natural-scene comparisons, sensitivity sweeps and quantitative analysis. |

Source is published at https://github.com/jush-ua/CSST-106. The repository URL is submitted to the class spreadsheet. Experiment outputs and analyses are under examples/; large datasets, environments and trained weights remain local.

## Image inputs and annotation provenance

Lab 02 uses a student-selected image. The bundled input.jpg is the public scikit-image Camera sample encoded as JPEG quality 95. Explicit file uploads or command-line paths select another image.

Lab 06 uses 257 unique COCO val2017 images, covering 100 positive images per class (bottle/cup/phone) and 30 backgrounds. Mixed scenes count toward multiple class quotas. Ten additional images are reserved for demonstration. COCO annotations were imported, visually reviewed and corrected where needed. Annotation-review metadata records the method and reviewer; imported annotations are distinguished from newly drawn boxes. All 22 contact sheets were inspected, with ambiguous examples checked at full resolution. The oversized phone box in image 380913 was corrected to the visible phone at the right.

The grouped split contains 180 training, 39 validation and 38 test images. Bottle/cup/phone instance counts are 129/130/81 for training, 21/31/23 for validation and 26/28/18 for test. Hash and perceptual checks found no duplicates crossing splits. Source IDs, URLs, licences, checksums, split assignments and review records are published in examples/lab06_custom/curation.json. COCO-pretrained weights have prior COCO exposure, so this is a classroom fine-tuning experiment rather than an independent benchmark.

## Measured results

- Lab 01: 512 x 512 x 3 input; 231,189 pixels have unequal channels. The selected pixel changes from BGR [7,14,19] to [0,0,255].
- Lab 02: decoded JPEG produces 8,300 Canny edge pixels and maximum raw Sobel magnitude 676.61 at the baseline settings.
- Lab 03: the controlled 100-155 intensity range expands to 0-255. Each case includes all 256 histogram bins.
- Lab 04: Hessian thresholds 100/400/1000 produce 1,548/737/389 source keypoints. At threshold 400, rotation gives 737 SURF versus 34 U-SURF accepted matches; translation gives 651 versus 659. Timings are single CPU observations, and accepted matches are not a geometric accuracy score.
- Lab 05: 1,288 faces, 1,850 features and 7 identities; 1,030 training and 258 test examples. The 150-component whitened PCA and balanced RBF SVM achieve accuracy 0.8488372093. The first 85 components retain 90% variance; 150 retain 0.9456145167. The 39 failed predictions were selected programmatically, with image-specific visual hypotheses in the PDF.
- Lab 06: precision 0.40811, recall 0.44485, mAP50 0.35573 and mAP50-95 0.25346. Epochs 1-68 used batch 4; after a CPU allocation failure, native resume retained optimizer state and completed epochs 69-100 at batch 2. The CSV contains all 100 epochs. Reserved-image analysis documents small-object misses and overlapping boxes.
- Lab 07: region-growing foreground IoU changes from 0.1859 at tolerance 8 to 0.6162 at 30; a background seed gives 0. Watershed marker fraction 0.35 merges the discs, while 0.65 and 0.85 produce two regions. DeepLab on the shared natural scene gives class mean IoU 0.6325 and pixel accuracy 0.9849. Binary foreground scores and semantic class scores measure different outputs.

## Verification

- Automated suite: 35 tests passed, covering image/annotation invariants, Lab 02 defaults, SURF CLI, UI state, video processing, checkpoint resume, frozen-data validation and evaluation-once behavior.
- Evidence integrity: 180 artifact hashes verified. Histogram totals match image sizes; 15 foreground IoUs were recalculated from saved masks; classical and semantic natural-scene inputs are pixel-identical.
- AppTest: all nine navigation entries load; processing actions for Labs 01, 02, 03, 04 and 07 pass. Completed analyses, downloadable figures, PCA PDF and Lab 06 missing-video feedback are available.
- Environment: application modules parse/compile, pip check reports no broken requirements, and the local server health endpoint returns HTTP 200.
- Native resume smoke test: a separate three-epoch synthetic fixture verifies optimizer retention and continuation to the original epoch target. Synthetic test metrics are separate from laboratory results.
- PCA PDF and comparison figures were visually inspected. Browser-based responsive and accessibility verification remains unperformed.

## Local environment and limitations

Python 3.11.9 and dependency versions are recorded in requirements.lock.txt. PyTorch runs on CPU; CUDA is unavailable. Visual Studio 2022 C++ Build Tools and Windows SDK are installed. The isolated OpenCV 4.12.0 build enables nonfree SURF; the main app retains its standard OpenCV environment.

The .venv, .surf-venv and data paths are directory junctions to D:\CSST106-build\runtime. Keep D: available. OpenCV source/build files are under D:\CSST106-build\opencv-src. Large binaries and caches are excluded from Git.

The PC has no webcam. Live camera capture remains untested. The ten reserved still images cover Lab 06's new-images-or-video inference step. A separate 12-frame Coffee-derived video fixture verifies uploaded-video processing: all 12 output frames decode at the source 12 FPS. This fixture measures the video pipeline and is not live camera footage.

Lab 06 training freezes the dataset and split metadata. Two optimizer-bearing checkpoints and an atomic progress pointer support interrupted runs. Pilots remain separate from completed models, and the held-out test evaluation runs only after 100 epochs. See LAB06_GUIDE.md for reproducible commands and examples/lab06_custom/DEMO_ANALYSIS.md for result interpretation.
