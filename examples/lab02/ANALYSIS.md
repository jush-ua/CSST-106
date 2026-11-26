# Lab 02 Sobel and Canny comparison

Source: scikit-image Camera; substitute because instructor input.jpg was not supplied. The supplied attachment names input.jpg, but no such image accompanied it. This run is explicitly a substitute demonstration. Reproduce on the actual file with `python complete_labs.py --labs 2 --edge-image PATH_TO_input.jpg`.

Both detectors use the same grayscale input and a 3 x 3 Gaussian blur with sigma 0. Sobel uses signed float64 x/y derivatives, kernel 3, and their Euclidean magnitude. Canny uses thresholds 100 and 200. The three-panel figure shows exactly original, Sobel and Canny in that order. The blurred image is saved separately. gx.npy, gy.npy and magnitude.npy preserve numerical values; only the displayed Sobel magnitude is normalized to 0-255.

Canny marks 8,212 pixels. The raw maximum Sobel magnitude is 675.42; 209,351 pixels remain nonzero in its normalized display. These counts are descriptive, not detector-accuracy scores. Sobel magnitude retains graded responses around the coat, tripod and textured ground; strong boundaries can occupy broad bands. Canny produces a binary, thinned map: the tripod and body outline stand out while much weak ground texture is suppressed. Some low-contrast boundaries disappear because they do not survive thresholding and hysteresis.

Sobel is useful when direction and gradient strength are needed. Canny is useful for compact boundaries and contour extraction. Smoothing reduces noisy gradients but also removes fine detail; threshold choice trades missed weak boundaries against extra responses. Neither edge map assigns object identities or fills object interiors.
