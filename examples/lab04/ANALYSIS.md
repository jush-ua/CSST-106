# Lab 04 measured SURF comparison

Public scikit-image Camera sample, 512 x 512. Matching uses two nearest neighbours and a 0.75 ratio test. SURF/SIFT use L2 distance; ORB uses Hamming distance. All rows use the same nonfree OpenCV 4.12.0 build.

Descriptors represent local appearance around detected keypoints so corresponding regions can be compared. The default SURF descriptor has 64 floating-point dimensions; SIFT has 128 and ORB has 32 bytes of binary tests. Accepted match counts measure ratio-test acceptance, not verified geometric correctness.

## Rotation

90 degrees clockwise.

| Detector | Hessian | Upright | Keypoints A/B | Matches | Detect + describe ms | Match ms |
|---|---:|---|---:|---:|---:|---:|
| SIFT | n/a | n/a | 792/799 | 762 | 242.59 | 8.37 |
| ORB | n/a | n/a | 1500/1500 | 1500 | 36.11 | 28.03 |
| SURF | 100 | False | 1548/1548 | 1548 | 249.14 | 23.76 |
| SURF | 100 | True | 1548/1548 | 58 | 113.84 | 25.07 |
| SURF | 400 | False | 737/737 | 737 | 167.24 | 6.25 |
| SURF | 400 | True | 737/737 | 34 | 93.36 | 9.01 |
| SURF | 1000 | False | 389/389 | 389 | 127.98 | 2.03 |
| SURF | 1000 | True | 389/389 | 15 | 79.59 | 1.79 |

Measured threshold sweep (A/B keypoints): 100: 1548/1548; 400: 737/737; 1000: 389/389.

## Upright Translation

translation x=20,y=15 pixels, black border.

| Detector | Hessian | Upright | Keypoints A/B | Matches | Detect + describe ms | Match ms |
|---|---:|---|---:|---:|---:|---:|
| SIFT | n/a | n/a | 792/716 | 678 | 206.79 | 7.84 |
| ORB | n/a | n/a | 1500/1500 | 1134 | 31.17 | 15.97 |
| SURF | 100 | False | 1548/1457 | 1326 | 233.02 | 21.43 |
| SURF | 100 | True | 1548/1457 | 1344 | 109.14 | 16.80 |
| SURF | 400 | False | 737/717 | 651 | 158.73 | 5.83 |
| SURF | 400 | True | 737/717 | 659 | 87.68 | 4.94 |
| SURF | 1000 | False | 389/393 | 344 | 131.73 | 2.06 |
| SURF | 1000 | True | 389/393 | 357 | 77.48 | 1.71 |

Measured threshold sweep (A/B keypoints): 100: 1548/1457; 400: 737/717; 1000: 389/393.

## Interpretation

Raising the Hessian threshold rejects weaker detector responses and generally leaves fewer keypoints. Inspect the measured counts above for this image. Fewer keypoints can reduce matching work while losing useful correspondences.

U-SURF skips dominant-orientation estimation. It is useful when camera/object orientation stays fixed, as in the translation case, but gives up descriptor rotation invariance. Compare its accepted matches at threshold 400 in both cases rather than treating every extra match as a correct correspondence.

SURF approximates Hessian responses with box filters/integral images and describes Haar-wavelet responses. SIFT uses scale-space extrema and gradient histograms; ORB uses oriented FAST and binary descriptors. The tables record this machine's measured detection/description and matching costs for each method. They are single CPU runs, include both images, may contain warm-up effects and do not establish a universal speed ranking. SIFT/ORB are capped at 1,500 keypoints by this implementation; SURF uses its Hessian threshold instead.
