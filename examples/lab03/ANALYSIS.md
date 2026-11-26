# Lab 03 Histogram equalization and contrast

Three reproducible public-sample cases cover restricted dynamic range, uneven illumination with noise, and colour luminance. Each case includes original/stretch/global equalization/CLAHE images, a four-panel comparison, histograms.png and all 256 histogram bins in histograms.csv. CLAHE uses clipLimit=2 and 8 x 8 tiles. The illumination and low-contrast images are explicitly derived fixtures.

| Case | Version | Range | Standard deviation | P95-P5 | Entropy bits |
|---|---|---|---:|---:|---:|
| low_contrast | original | 100-155 | 15.89 | 43.00 | 5.05 |
| low_contrast | stretched | 0-255 | 73.65 | 199.00 | 5.05 |
| low_contrast | equalized | 0-255 | 73.74 | 230.00 | 5.03 |
| low_contrast | clahe | 68-195 | 22.76 | 69.00 | 6.27 |
| uneven_illumination | original | 0-232 | 60.48 | 181.00 | 7.41 |
| uneven_illumination | stretched | 0-255 | 66.49 | 199.00 | 7.41 |
| uneven_illumination | equalized | 0-255 | 74.28 | 231.00 | 7.25 |
| uneven_illumination | clahe | 1-243 | 58.22 | 177.00 | 7.61 |
| colour_luminance | original | 0-255 | 58.12 | 194.00 | 7.66 |
| colour_luminance | stretched | 0-255 | 58.12 | 194.00 | 7.66 |
| colour_luminance | equalized | 0-255 | 73.48 | 228.00 | 7.44 |
| colour_luminance | clahe | 0-255 | 59.24 | 201.00 | 7.80 |

## Observations

The low-contrast original occupies only 100-155. Stretching maps these endpoints to 0 and 255, making the coat, tripod and background more distinct without inventing new intensity levels. Global equalization redistributes the populated bins through the cumulative histogram, while CLAHE changes local neighbourhood contrast. In the uneven-light case, global mapping cannot compensate independently for the dark left and brighter right. CLAHE can reveal local structure but also makes grain in smooth regions more noticeable. Its clipping limits this effect; it does not remove the added noise. Greater spread or entropy alone does not establish better visual quality.

The colour case applies each mapping only to Y, then reconstructs colour with the original U/V channels. Processing B/G/R independently could distort colour relationships. Conversion and clipping can still alter reconstructed colours; luminance processing does not guarantee perfect colour preservation.

## Questions and conclusion

Histogram equalization remaps pixel values using their cumulative frequency, approximately distributing occupied intensities across the available output range. It helps reveal details compressed into a narrow brightness range, such as inspection surfaces or poorly exposed photographs. It cannot recover clipped highlights or missing information. CLAHE computes clipped local histograms and interpolates mappings between tiles, addressing local illumination differences while limiting contrast amplification. Stretching is the simpler linear endpoint mapping. Global equalization uses one nonlinear mapping for the entire image; CLAHE trades extra local computation and parameter choices for spatial adaptation. Choose by visibility of relevant structures and artifacts, rather than maximizing a histogram statistic.
