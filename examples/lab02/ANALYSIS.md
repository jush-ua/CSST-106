# Lab 02 Sobel and Canny comparison

Source: Packaged input.jpg (student-selected scikit-image Camera image). A real JPEG input.jpg and its source/checksum are bundled with this lab. The bundled file is a student-selected image, as permitted by the activity. Measurements are recomputed from the decoded JPEG, including its compression effects. Override it with `python complete_labs.py --labs 2 --edge-image PATH_TO_input.jpg`.

Both detectors use the same grayscale input and a 3 x 3 Gaussian blur with sigma 0. Sobel uses signed float64 x/y derivatives, kernel 3, and their Euclidean magnitude. Canny uses thresholds 100 and 200. The three-panel figure shows exactly original, Sobel and Canny in that order. The blurred image is saved separately. gx.npy, gy.npy and magnitude.npy preserve numerical values; only the displayed Sobel magnitude is normalized to 0-255.

Canny marks 8,300 pixels. The raw maximum Sobel magnitude is 676.61; 214,717 pixels remain nonzero in its normalized display. These counts are descriptive, not detector-accuracy scores. Sobel magnitude retains graded responses; strong boundaries can occupy broad bands. Canny produces a binary, thinned map. In the packaged Camera image, the tripod and body outline stand out while much weak ground texture is suppressed. Some low-contrast boundaries disappear because they do not survive thresholding and hysteresis.

Sobel is useful when direction and gradient strength are needed. Canny is useful for compact boundaries and contour extraction. Smoothing reduces noisy gradients but also removes fine detail; threshold choice trades missed weak boundaries against extra responses. Neither edge map assigns object identities or fills object interiors.
