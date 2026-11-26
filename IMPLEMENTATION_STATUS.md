# Implementation and verification status

Verified 26 November 2026. This is an implementation with real sample runs, not a declaration that every assignment requirement has been completed.

| Component | Evidence | Remaining work |
|---|---|---|
| Shared Web UI | All 9 navigation entries load without AppTest exceptions; classical action checks pass; HTTP health returns 200 | Visual browser service unavailable, so responsive/visual QA is not claimed |
| Lab 01 | Actual colour Astronaut run, BGR values before/after editing, grayscale, crop, resize and rotation; six-panel figure and measured analysis exported | None for requested digital work |
| Lab 02 | Bundled real JPEG is the default in UI/CLI/generator; JPEG-decoded gradients, exact three-panel figure, measured differences and conclusion regenerated | Actual input.jpg is bundled with explicit replacement provenance; Jeremy's original remains unavailable and can override it later |
| Lab 03 | Low-contrast, uneven illumination/noise and colour-YUV cases; comparison figures, histogram PNG/CSV, measurements, observations and question answers saved | None for requested digital work |
| Lab 04 | Actual nonfree SURF/U-SURF threshold sweeps (100/400/1000), SIFT/ORB comparisons, rich keypoints and matches on rotation and translation pairs; see examples/lab04/ANALYSIS.md | No missing requested implementation; standalone command and UI share the verified worker. Timings remain single-run CPU observations |
| Lab 05 | Real LFW run, classification report, variance, eigenfaces, failure gallery and visually verified 3-page digital PDF | No missing computational requirement; user can refine interpretations through the PDF editor |
| Lab 06 | Pretrained inference/video verified; phone capture guide, session progress/contact sheets, frozen-data pilot/full/resume workflow, report/demo/bundle tooling implemented | Personal bottle/mug/phone photos and their annotations are absent; custom 100-epoch run and held-out metrics have NOT been executed. Uploaded-video inference passed: 12/12 frames decoded at 12 FPS. Live webcam verification is unavailable: user confirms this computer has no webcam |
| Lab 07 | Five classical methods on four conditions, seed/marker sweeps, aligned foreground scores and timings; DeepLab on the same natural scene; figures and measured analysis saved, plus prior five-VOC evaluation | No missing requested digital experiment; whole-app visual browser QA is not claimed |
| Repository | Published to https://github.com/jush-ua/CSST-106 on main using local Git credentials; connected-app 403 was resolved through local Git | Repository URL already submitted to the classroom spreadsheet, confirmed by the user; no new entry needed |

## Completed demonstrations and analyses

`complete_labs.py` regenerates Labs 01-03 and 07 under `examples/`. Every lab has an ANALYSIS.md, figures and measured manifests; all comparison and histogram figures were visually inspected. The instructor input.jpg was not supplied. Lab 02 now includes a disclosed Camera-derived JPEG and source/checksum metadata; the actual JPEG is decoded for every regenerated result. The original/Sobel/Canny panels now occupy the first UI row. The application exposes completed analyses and downloadable figures without requiring an experiment rerun.

- Lab 01: 512 x 512 x 3 colour input; 231,189 pixels with unequal channels; edited pixel [7,14,19] -> [0,0,255].
- Lab 03: controlled low-contrast range 100-155 expands to 0-255; three cases include full 256-bin histogram CSVs.
- Lab 07: controlled touching discs, an illumination/noise variant, Camera texture and one natural VOC scene. Known masks support foreground-union IoU/Dice; unknown Camera truth is never invented.
- Region tolerance sweep: foreground IoU 0.1859 at tolerance 8, 0.6162 at 30; background seed gives 0. Watershed marker fraction 0.35 merges the discs into one region; 0.65 and 0.85 produce two regions.
- Natural-scene DeepLab class mean IoU 0.6325 and pixel accuracy 0.9849. Foreground scores for classical methods and semantic class scores are kept distinct.
- Timings are single CPU observations with preprocessing recorded separately; no universal performance ranking is asserted.

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

- `pytest tests -q`: 32 passed, including Lab 02 default/provenance and UI, Lab 06 pilot OOM fallback, pause/resume, retained checkpoint handoff, changed-data refusal, evaluation-once guard, private/public artifact separation, and the previous image/video/SURF checks.
- Evidence integrity: 171 artifact hashes verified (including the new Lab 02 JPEG/provenance); histogram bin totals match image sizes; 15 foreground IoUs independently recalculated from saved masks; classical and semantic natural-scene inputs are pixel-identical. Figure export margins checked after fixing one clipped semantic title.
- `pip check`: no broken requirements.
- All application modules parse/compile.
- AppTest: navigation through all modules; processing actions for Labs 01,02,03,04,07; source-change invalidation regression.
- Completed reference AppTest: Labs 01/02/03/07 expose 1/1/6/7 downloadable figures respectively; no exception/error. Lab 04 shared-worker UI loads all eight rows and all images.
- Lab 04 AppTest after build: eight comparison rows, six SURF sweep rows, all images loaded, OpenCV 4.12.0 on every row, no exception/error. Lab 06 missing-video warning passes.
- Scientific image and annotation tests: shape/order, no in-place edits, Sobel/Canny baseline, constant contrast, empty descriptors, connectivity, tolerance equality, void labels, mismatched masks, session isolation and bounds.
- Local preview health: 200. Two CUA service attempts failed before browser interaction, so no browser screenshot/contrast/responsive verification is represented as completed.
- Static finish review: four material state/copy findings fixed and confirmed resolved; this is not a whole-surface visual approval.

## Media evidence

`examples/media_verification` contains a public Coffee-derived 12-frame fixture, actual YOLO detections video and machine-readable verification. The same processing function is used by the UI. No camera footage was captured; Windows and OpenCV found no camera, and the user confirmed there is no webcam.

## Environment constraints

Python 3.11.9, installed packages pinned in requirements.lock.txt. PyTorch reports CPU-only; CUDA unavailable in this environment. Visual Studio 2022 C++ Build Tools 17.14.45 and Windows SDK 10.0.26100 installed successfully. Compiler MSVC 19.44.35230.0 is on D:. The isolated OpenCV 4.12.0 build completed with nonfree algorithms enabled and SURF detector construction verified. Standard OpenCV remains the main application build; setup-surf.ps1 reproduces the isolated source build.

The local `.venv`, `.surf-venv` and `data` paths are directory junctions to `D:\CSST106-build\runtime` to preserve C: space. Keep that drive connected. OpenCV sources and build files are in `D:\CSST106-build\opencv-src`; large binaries and caches are excluded from Git.

No personal training scores, camera footage, handwritten submissions or paper reports have been fabricated. The user confirmed their previous spreadsheet submission; this implementation has not posted another entry.

## Lab 06 preparation and remaining dependency

`LAB06_GUIDE.md` documents phone capture (330-image target across ten sessions), annotation, pilot timing, CPU training, resume, phone-video comparison and private/public deliverables. The runtime will not start with missing personal data. Pilots cannot enter the completed-model selector; incomplete full runs cannot evaluate the test set or claim completion. A changed frozen dataset or checkpoint hash blocks resume. Two optimizer-bearing checkpoints are retained with an atomic progress pointer, limiting disk use and retaining the previous valid pair during interrupted writes.

The real installed Ultralytics library passed `tests/native_yolo_resume_smoke.py`, a separate synthetic checkpoint smoke test: optimizer state saved after epoch index 0; native resume executed only indices 1 and 2, giving three total epochs. That smoke test uses 64-pixel synthetic fixtures solely to verify library mechanics. It is not the required personal-data experiment, which remains 640 pixels and 100 epochs. Its files and meaningless synthetic accuracy values remain in ignored runs, not in public lab results.

Still pending: personal image capture and annotation, pilot timing on that actual dataset, full 100-epoch training, held-out metrics, new-image/video demonstration and qualitative error review. The PC has no webcam; the selected alternative is recorded phone video, with acceptance of that substitution pending Jeremy.
