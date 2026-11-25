# Implementation and verification status

Verified 25 November 2026. This is an implementation with real sample runs, not a declaration that every assignment requirement has been completed.

| Component | Evidence | Remaining work |
|---|---|---|
| Shared Web UI | All 9 navigation entries load without AppTest exceptions; classical action checks pass; HTTP health returns 200 | Visual browser service unavailable, so responsive/visual QA is not claimed |
| Lab 01 | Sample pixel edit, grayscale, crop, resize and clockwise rotation exported | Use additional personal/instructor inputs as desired |
| Lab 02 | Baseline Sobel/Canny run and signed gradient arrays saved; source-operation test passes | Instructor's separately referenced input.jpg was not attached |
| Lab 03 | Global equalization, min-max stretch and CLAHE sample run; colour/YUV and constant-image tests pass | None for implementation |
| Lab 04 | ORB/SIFT matching sample run; empty-descriptor tests pass | Actual SURF/U-SURF sweep awaits Visual Studio C++ tools and the isolated nonfree build |
| Lab 05 | Real LFW run, classification report, variance, eigenfaces, failure gallery and visually verified 3-page digital PDF | No missing computational requirement; user can refine interpretations through the PDF editor |
| Lab 06 | Pretrained YOLO11 detection executed on public Coffee sample; dataset editor/validation/training pipeline implemented | Personal bottle/mug/phone photos and their annotations are absent; custom 100-epoch run and held-out metrics have NOT been executed. Webcam and uploaded-video paths remain unverified on hardware |
| Lab 07 | Classical sample run plus DeepLab inference and aligned scoring for five verified VOC validation pairs | Browser visual verification unavailable; exact sample metrics are in examples |
| Repository | Published to https://github.com/jush-ua/CSST-106 on main using local Git credentials; connected-app 403 was resolved through local Git | No classroom spreadsheet entry was submitted |

## Actual PCA results

- 1,288 faces, 1,850 features, 7 identities; training 1,030 and test 258.
- Required PCA settings: 150 randomized components, whitening, seed 42; SVM: balanced RBF.
- Held-out accuracy: 0.8488372093; 39 failed predictions selected programmatically.
- Variance at 150 components: 0.9456145167; first 85 components reach 90%.
- PDF figures, measured statistics and image-specific hypotheses were inspected. Causes of errors remain hypotheses, not proven findings.

## Verification

- `pytest tests -q`: 17 passed.
- `pip check`: no broken requirements.
- All application modules parse/compile.
- AppTest: navigation through all modules; processing actions for Labs 01,02,03,04,07; source-change invalidation regression.
- Scientific image and annotation tests: shape/order, no in-place edits, Sobel/Canny baseline, constant contrast, empty descriptors, connectivity, tolerance equality, void labels, mismatched masks, session isolation and bounds.
- Local preview health: 200. Two CUA service attempts failed before browser interaction, so no browser screenshot/contrast/responsive verification is represented as completed.
- Static finish review: four material state/copy findings fixed and confirmed resolved; this is not a whole-surface visual approval.

## Environment constraints

Python 3.11.9, installed packages pinned in requirements.lock.txt. PyTorch reports CPU-only; CUDA unavailable in this environment. Visual Studio C++ Build Tools were not found. Standard OpenCV does not supply the required nonfree SURF capability; setup-surf.ps1 contains the isolated source-build procedure.

No custom training scores, personal photos, handwritten submissions, paper reports or classroom spreadsheet submissions have been fabricated or posted.
