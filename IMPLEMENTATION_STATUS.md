# Implementation and verification status

Verified 26 November 2026 against all seven original DOCX attachments. Excluding hard-copy reports and the webcam requirement, no unfinished computational work was found. Lab 02 is complete using its bundled student-selected image; no instructor-provided input image is expected. Hard-copy reports and live webcam testing are outside this completion scope.

| Component | Evidence | Remaining work |
|---|---|---|
| Shared Web UI | All 9 navigation entries load without AppTest exceptions; classical action checks pass; HTTP health returns 200 | Visual browser service unavailable, so responsive/visual QA is not claimed |
| Lab 01 | Actual colour Astronaut run, BGR values before/after editing, grayscale, crop, resize and rotation; six-panel figure and measured analysis exported | None for requested digital work |
| Lab 02 | Bundled real JPEG is the default in UI/CLI/generator; JPEG-decoded gradients, exact three-panel figure, measured differences and conclusion regenerated | None for requested digital work; input.jpg is student-selected, with source/checksum provenance and optional overrides |
| Lab 03 | Low-contrast, uneven illumination/noise and colour-YUV cases; comparison figures, histogram PNG/CSV, measurements, observations and question answers saved | None for requested digital work |
| Lab 04 | Actual nonfree SURF/U-SURF threshold sweeps (100/400/1000), SIFT/ORB comparisons, rich keypoints and matches on rotation and translation pairs; see examples/lab04/ANALYSIS.md | No missing requested implementation; standalone command and UI share the verified worker. Timings remain single-run CPU observations |
| Lab 05 | Real LFW run, classification report, variance, eigenfaces, failure gallery and visually verified 3-page digital PDF | No missing computational requirement; user can refine interpretations through the PDF editor |
| Lab 06 | Reviewed/corrected public COCO collection; 100 training epochs at 640 pixels; one held-out evaluation; ten reserved-image comparisons visually reviewed; metrics, charts and error analysis published; trained weights and image outputs in the private bundle | None within the final review scope. Part 4 permits new images or video, so another video is unnecessary. Public image sources and imported annotation provenance are documented. Live webcam work remains unperformed and excluded |
| Lab 07 | Five classical methods on four conditions, seed/marker sweeps, aligned foreground scores and timings; DeepLab on the same natural scene; figures and measured analysis saved, plus prior five-VOC evaluation | No missing requested digital experiment; whole-app visual browser QA is not claimed |
| Repository | Published to https://github.com/jush-ua/CSST-106 on main | Repository URL already submitted to the classroom spreadsheet as requested by the instructor |

## Completion scope

The implementation was checked against all seven source attachments. Labs 01, 03, 04, 05, 06 and 07 have no unfinished computational requirements within the requested scope. Lab 05's required digital PDF is included, and its variance, visual interpretation, whitening, failure analysis and all five follow-up answers were checked. Lab 04's standalone command shares the verified nonfree SURF worker.

Lab 02 is complete and demonstrated with the bundled Camera-derived `input.jpg`. Students choose their input image for this activity. Source/checksum metadata remains available, and students can choose another input without changing the implementation.

Lab 06 uses a curated public COCO collection with imported, visually reviewed/corrected annotations. The attachment does not explicitly require personally photographed images. Its new-image-or-video inference step is satisfied by the ten reserved still images; no separate recorded-video demonstration is outstanding for that step. Live webcam testing is a separate hardware-dependent step and remains unperformed.

Final evidence checks verified 180 artifact hashes, confirmed the 100-epoch result and completed held-out evaluation, and checked that the private Lab 06 ZIP contains best.pt and demonstration outputs. The repository URL has already been submitted to the classroom spreadsheet. Hard-copy reports and webcam work are excluded; browser visual QA remains an engineering verification limitation rather than an uncompleted lab experiment.

## Completed demonstrations and analyses

`complete_labs.py` regenerates Labs 01-03 and 07 under `examples/`. Every lab has an ANALYSIS.md, figures and measured manifests; all comparison and histogram figures were visually inspected. Lab 02 uses a student-selected Camera-derived JPEG and source/checksum metadata; the actual JPEG is decoded for every regenerated result. The original/Sobel/Canny panels now occupy the first UI row. The application exposes completed analyses and downloadable figures without requiring an experiment rerun.

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

- `pytest tests -q`: 35 passed, including Lab 02 default/provenance and UI, Lab 06 pilot OOM fallback, pause/resume, retained checkpoint handoff, changed-data refusal, evaluation-once guard, private/public artifact separation, and the previous image/video/SURF checks.
- Evidence integrity: 180 artifact hashes verified (including the Lab 02 JPEG/provenance and completed Lab 06 public results); histogram bin totals match image sizes; 15 foreground IoUs independently recalculated from saved masks; classical and semantic natural-scene inputs are pixel-identical. Figure export margins checked after fixing one clipped semantic title.
- `pip check`: no broken requirements.
- All application modules parse/compile.
- AppTest: navigation through all modules; processing actions for Labs 01,02,03,04,07; source-change invalidation regression.
- Completed reference AppTest: Labs 01/02/03/07 expose 1/1/6/7 downloadable figures respectively; no exception/error. Lab 04 shared-worker UI loads all eight rows and all images.
- Lab 04 AppTest after build: eight comparison rows, six SURF sweep rows, all images loaded, OpenCV 4.12.0 on every row, no exception/error. Lab 06 missing-video warning passes.
- Scientific image and annotation tests: shape/order, no in-place edits, Sobel/Canny baseline, constant contrast, empty descriptors, connectivity, tolerance equality, void labels, mismatched masks, session isolation and bounds.
- Local preview health: 200. Two CUA service attempts failed before browser interaction, so no browser screenshot/contrast/responsive verification is represented as completed.
- Static finish review: four material state/copy findings fixed and confirmed resolved; this is not a whole-surface visual approval.

## Media evidence

`examples/media_verification` contains a public Coffee-derived 12-frame fixture, actual YOLO detections video and machine-readable verification. The same processing function is used by the UI. No camera footage was captured; this PC has no webcam, and Windows and OpenCV detected no camera.

## Environment constraints

Python 3.11.9, installed packages pinned in requirements.lock.txt. PyTorch reports CPU-only; CUDA unavailable in this environment. Visual Studio 2022 C++ Build Tools 17.14.45 and Windows SDK 10.0.26100 installed successfully. Compiler MSVC 19.44.35230.0 is on D:. The isolated OpenCV 4.12.0 build completed with nonfree algorithms enabled and SURF detector construction verified. Standard OpenCV remains the main application build; setup-surf.ps1 reproduces the isolated source build.

The local `.venv`, `.surf-venv` and `data` paths are directory junctions to `D:\CSST106-build\runtime` to preserve C: space. Keep that drive connected. OpenCV sources and build files are in `D:\CSST106-build\opencv-src`; large binaries and caches are excluded from Git.

Training metrics are measured experiment results. No live camera footage or hard-copy report is included. The repository URL is already submitted to the class spreadsheet.

## Lab 06 completion and excluded webcam requirement

`LAB06_GUIDE.md` now documents curated COCO import/review, source grouping, pilot timing, CPU training, resume, separate still-image demonstration and private/public deliverables. Personal capture remains optional; the selected collection is public. Pilots cannot enter the completed-model selector; incomplete full runs cannot evaluate the test set or claim completion. A changed frozen dataset or checkpoint hash blocks resume. Two optimizer-bearing checkpoints are retained with an atomic progress pointer, limiting disk use and retaining the previous valid pair during interrupted writes.

The real installed Ultralytics library passed `tests/native_yolo_resume_smoke.py`, a separate synthetic checkpoint smoke test: optimizer state saved after epoch index 0; native resume executed only indices 1 and 2, giving three total epochs. That smoke test uses 64-pixel synthetic fixtures solely to verify library mechanics. It is not the actual curated-data experiment, which remains 640 pixels and 100 epochs. Its files and meaningless synthetic accuracy values remain in ignored runs, not in public lab results.

The experiment uses a curated public COCO collection. Import/review/correction, full training, held-out evaluation and reserved-image review are complete. The PC has no webcam; no still-image or synthetic-video output is represented as live capture. A recorded clip can be processed separately through the video inference feature; it is distinct from live webcam testing.

## Curated COCO collection

The curated collection uses source categories bottle/cup/cell phone and an annotation import, review and correction workflow. `curate_coco.py` selected 257 unique COCO val2017 images, covering 100 positive images per class and 30 backgrounds; mixed scenes satisfy multiple quotas. Ten additional images are reserved for demonstration. All 22 contact sheets were visually inspected, ambiguous examples were opened at full resolution, and an oversized phone source box in image 380913 was corrected to the visible phone at the right. Review metadata identifies assistant inspection rather than claiming personally drawn labels.

The grouped split is 180/39/38 images for training/validation/test, with bottle/cup/phone instance counts respectively 129/130/81, 21/31/23 and 26/28/18. Exact hashes and the perceptual near-duplicate check found no duplicates crossing splits. Source licence/URL/checksum and split records are in examples/lab06_custom/curation.json. COCO-pretrained weights have prior COCO exposure; this exercise is not an independent dataset benchmark. Training completed 100 epochs and one held-out evaluation. Precision/recall/mAP50/mAP50–95 are 0.40811/0.44485/0.35573/0.25346. Epochs 1–68 used batch 4; after a CPU allocation failure, native resume completed epochs 69–100 at batch 2. Ten reserved-image comparisons were generated and visually reviewed; source records, metrics, curves, confusion charts and error analysis are published in examples/lab06_custom. The weights and image outputs are bundled privately.
