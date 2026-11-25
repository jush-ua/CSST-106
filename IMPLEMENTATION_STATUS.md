# Implementation and verification status

Verified 25 November 2026. This is an implementation with real sample runs, not a declaration that every assignment requirement has been completed.

| Component | Evidence | Remaining work |
|---|---|---|
| Shared Web UI | All 9 navigation entries load without AppTest exceptions; classical action checks pass; HTTP health returns 200 | Visual browser service unavailable, so responsive/visual QA is not claimed |
| Lab 01 | Sample pixel edit, grayscale, crop, resize and clockwise rotation exported | Use additional personal/instructor inputs as desired |
| Lab 02 | Baseline Sobel/Canny run and signed gradient arrays saved; source-operation test passes | Instructor's separately referenced input.jpg was not attached |
| Lab 03 | Global equalization, min-max stretch and CLAHE sample run; colour/YUV and constant-image tests pass | None for implementation |
| Lab 04 | Actual nonfree SURF/U-SURF threshold sweeps (100/400/1000), SIFT/ORB comparisons, rich keypoints and matches on rotation and translation pairs; see examples/lab04/ANALYSIS.md | None for implementation; timings are single-run CPU observations |
| Lab 05 | Real LFW run, classification report, variance, eigenfaces, failure gallery and visually verified 3-page digital PDF | No missing computational requirement; user can refine interpretations through the PDF editor |
| Lab 06 | Pretrained YOLO11 detection executed on public Coffee sample; dataset editor/validation/training pipeline implemented | Personal bottle/mug/phone photos and their annotations are absent; custom 100-epoch run and held-out metrics have NOT been executed. Uploaded-video inference passed: 12/12 frames decoded at 12 FPS. Live webcam verification is unavailable: user confirms this computer has no webcam |
| Lab 07 | Classical sample run plus DeepLab inference and aligned scoring for five verified VOC validation pairs | Browser visual verification unavailable; exact sample metrics are in examples |
| Repository | Published to https://github.com/jush-ua/CSST-106 on main using local Git credentials; connected-app 403 was resolved through local Git | No classroom spreadsheet entry was submitted |

## Actual SURF results

- Camera input: 512 x 512; threshold 100/400/1000 gives 1,548/737/389 source keypoints.
- At threshold 400, 90-degree rotation: SURF 737 accepted matches, U-SURF 34.
- Translation x=20/y=15: SURF 651 accepted matches, U-SURF 659; detection/description 158.73/87.68 ms respectively.
- SURF, SIFT and ORB run on the same pairs in the same OpenCV 4.12.0 build. All 16 comparison records and their PNG artifacts are saved. Accepted matches are not a geometric accuracy score; timings are single-run measurements.

## Actual PCA results

- 1,288 faces, 1,850 features, 7 identities; training 1,030 and test 258.
- Required PCA settings: 150 randomized components, whitening, seed 42; SVM: balanced RBF.
- Held-out accuracy: 0.8488372093; 39 failed predictions selected programmatically.
- Variance at 150 components: 0.9456145167; first 85 components reach 90%.
- PDF figures, measured statistics and image-specific hypotheses were inspected. Causes of errors remain hypotheses, not proven findings.

## Verification

- `pytest tests -q`: 20 passed, including video encode/decode, unreadable input and overwrite protection.
- `pip check`: no broken requirements.
- All application modules parse/compile.
- AppTest: navigation through all modules; processing actions for Labs 01,02,03,04,07; source-change invalidation regression.
- Lab 04 AppTest after build: eight comparison rows, six SURF sweep rows, all images loaded, OpenCV 4.12.0 on every row, no exception/error. Lab 06 missing-video warning passes.
- Scientific image and annotation tests: shape/order, no in-place edits, Sobel/Canny baseline, constant contrast, empty descriptors, connectivity, tolerance equality, void labels, mismatched masks, session isolation and bounds.
- Local preview health: 200. Two CUA service attempts failed before browser interaction, so no browser screenshot/contrast/responsive verification is represented as completed.
- Static finish review: four material state/copy findings fixed and confirmed resolved; this is not a whole-surface visual approval.

## Media evidence

`examples/media_verification` contains a public Coffee-derived 12-frame fixture, actual YOLO detections video and machine-readable verification. The same processing function is used by the UI. No camera footage was captured; Windows and OpenCV found no camera, and the user confirmed there is no webcam.

## Environment constraints

Python 3.11.9, installed packages pinned in requirements.lock.txt. PyTorch reports CPU-only; CUDA unavailable in this environment. Visual Studio 2022 C++ Build Tools 17.14.45 and Windows SDK 10.0.26100 installed successfully. Compiler MSVC 19.44.35230.0 is on D:. The isolated OpenCV 4.12.0 build completed with nonfree algorithms enabled and SURF detector construction verified. Standard OpenCV remains the main application build; setup-surf.ps1 reproduces the isolated source build.

The local `.venv`, `.surf-venv` and `data` paths are directory junctions to `D:\CSST106-build\runtime` to preserve C: space. Keep that drive connected. OpenCV sources and build files are in `D:\CSST106-build\opencv-src`; large binaries and caches are excluded from Git.

No custom training scores, personal photos, handwritten submissions, paper reports or classroom spreadsheet submissions have been fabricated or posted.
