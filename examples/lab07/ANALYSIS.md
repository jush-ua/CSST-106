# Lab 07 Segmentation comparison

All five classical methods use the same input within each case. The controlled discs and their illumination variant have exact union masks; they are test fixtures, not personal photographs. Camera provides complex texture without invented ground truth. The natural scene is a real VOC validation pair, also passed to pretrained DeepLabV3-ResNet50. The existing five-image semantic evaluation remains in examples/semantic/five_image_scores.json.

| Case | Method | Foreground IoU | Foreground Dice | Stage ms |
|---|---|---:|---:|---:|
| touching_objects | otsu | 0.9997 | 0.9999 | 0.42 |
| touching_objects | adaptive | 0.1548 | 0.2681 | 0.43 |
| touching_objects | region | 1.0000 | 1.0000 | 47.11 |
| touching_objects | edge_regions | 0.9742 | 0.9869 | 1.05 |
| touching_objects | watershed | 0.9556 | 0.9773 | 9.10 |
| uneven_illumination | otsu | 0.9894 | 0.9947 | 0.05 |
| uneven_illumination | adaptive | 0.1536 | 0.2663 | 0.19 |
| uneven_illumination | region | 0.6162 | 0.7626 | 19.99 |
| uneven_illumination | edge_regions | 0.9712 | 0.9854 | 0.73 |
| uneven_illumination | watershed | 0.9697 | 0.9846 | 9.27 |
| texture | otsu | n/a | n/a | 0.09 |
| texture | adaptive | n/a | n/a | 0.28 |
| texture | region | n/a | n/a | 8.27 |
| texture | edge_regions | n/a | n/a | 2.51 |
| texture | watershed | n/a | n/a | 9.60 |
| natural_scene | otsu | 0.1150 | 0.2062 | 0.15 |
| natural_scene | adaptive | 0.1139 | 0.2045 | 0.61 |
| natural_scene | region | 0.0535 | 0.1016 | 408.42 |
| natural_scene | edge_regions | 0.3311 | 0.4974 | 8.99 |
| natural_scene | watershed | 0.1149 | 0.2061 | 39.24 |

Binary scores measure foreground union only, not object identity or separation. Watershed uses labels >1 as foreground and counts boundary pixels as excluded. VOC void=255 is ignored. Classical methods are not trained to distinguish VOC categories; these objectness scores must not be equated with semantic mean IoU. Stage timings exclude shared grayscale/blur preprocessing and plotting; that preprocessing and total time are recorded separately. Single CPU timings are descriptive, not a controlled benchmark.

## Observations from the displayed cases

On the clean discs, Otsu foreground IoU is 0.9997 and seeded growing is 1.0000. Both recover the union, but neither separates the two instances. Adaptive thresholding with positive C makes much of the uniform background white and produces a dark ring around the objects, explaining its poor foreground score here. It is not automatically better than global thresholding.

With the illumination ramp and noise, Otsu still achieves 0.9894 because foreground and background remain largely distinguishable. The region mask truncates inside the right-hand disc as brightness departs from the seed; the sensitivity figures show how increasing tolerance recovers more of the object without fully reaching its brighter side.

On Camera, the central seed selects only a small connected patch. Bright-foreground thresholding groups sky and ground while leaving much of the dark coat black. Filled edge contours recover some tripod and face regions but also follow texture; watershed groups broad brightness areas, not the person. There is no annotated Camera mask, so no quantitative accuracy is claimed for this case.

In the aircraft scene, edge filling has foreground IoU 0.3311, but includes foreground equipment and background regions. The centre seed grows mainly into the sky, and bright-foreground Otsu also selects much sky. This is a polarity/appearance mismatch with the annotated aircraft, not a claim that thresholding universally fails. DeepLab uses learned context to cover the fuselage, wings and tail much more coherently, though boundaries and small predicted fragments still need inspection.

## Seed and watershed sensitivity

- object_tol8: seed=(96, 96), tolerance=8, 1290 pixels, foreground IoU=0.1859.
- object_tol30: seed=(96, 96), tolerance=30, 4276 pixels, foreground IoU=0.6162.
- background_tol30: seed=(10, 10), tolerance=30, 42212 pixels, foreground IoU=0.0000.
- Watershed fraction 0.35: 1 initial components, 1 output object regions.
- Watershed fraction 0.65: 2 initial components, 2 output object regions.
- Watershed fraction 0.85: 2 initial components, 2 output object regions.

Region growing accepts connected pixels within tolerance of the original seed intensity. With uneven light a small tolerance can stop inside one object, while a larger tolerance expands further; a background seed selects a different region entirely. Automatic watershed depends on foreground polarity and marker placement. Connected markers can leave touching objects merged; more markers can split them, and excessive or noisy markers can over-segment a single object. Count and inspect regions as well as foreground overlap. A high binary IoU does not prove the two discs were separated.

## Natural scene semantic result

DeepLab labels the same 500 x 366 scene in 11.07 seconds on CPU. Its class mean IoU is 0.6325, and pixel accuracy is 0.9849. Predicted labels: __background__, aeroplane, person. The palette, class-ID mask and overlay are saved with a legend in result.json. High pixel accuracy can be dominated by background; class IoUs expose small or spurious classes. The model recognizes a fixed vocabulary and may miss objects or assign an incorrect class at boundaries.

## Use cases limitations and conclusion

Thresholding is fast and appropriate for high-contrast foreground/background separation. Otsu uses one global intensity split; adaptive thresholding handles local brightness changes but its block size and constant can turn texture or background into foreground. Compare the controlled clean and uneven cases with identical geometry to isolate illumination effects.

Region methods use connectivity and intensity similarity. Seeded growing is useful for a known homogeneous region but can leak into similar background or stop at a lighting gradient. Watershed is useful for touching objects when markers represent the intended objects; its labels have no semantic meaning.

Edges locate intensity transitions. Closing and filling contours can recover regions with clear boundaries, but noise makes extra contours and gaps can connect object and background. Camera texture demonstrates why many edge responses are not necessarily useful object borders. Compare the Canny map with the filled edge regions and watershed boundaries rather than treating every contour as an object.

Semantic segmentation supplies learned per-pixel categories and scene context instead of brightness groups. It requires pretrained weights, tensor preprocessing, a compatible class vocabulary and substantially more memory/computation here. Thin structures, unusual appearance and boundaries can still be wrong. No semantic training is required for this activity.

Thresholds and fixed-size local filtering scan the image; region growing can visit the entire connected image and this Python queue has noticeable overhead. Watershed adds distance, marker and flooding work. Contour extraction depends on the number and length of boundaries. DeepLab executes many convolutional layers, with runtime and memory depending on input size, model and device. The measured tables include single-run effects and must not be read as universal speed rankings. Mask alignment, polarity, seed coordinates, void labels and the difference between instance labels and class labels are key implementation challenges. The results support choosing the method by scene and intended output, rather than declaring one method best for every image.
