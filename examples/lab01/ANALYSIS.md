# Lab 01 Colour image operations

The public Astronaut input is 512 pixels high by 512 wide with three BGR channels. 231,189 pixels have unequal channel values, so this is a genuine colour demonstration. Each uint8 channel stores a value from 0 to 255. OpenCV orders colour channels blue, green, red; Matplotlib display converts back to RGB. Grayscale is a two-dimensional intensity array and discards colour distinctions.

At row 256, column 256, the BGR value changes from [7, 14, 19] to [0, 0, 255] (red). Only the edited copy changes. The enlarged nearest-neighbour patch makes this one-pixel operation visible. The crop begins at x=64, y=48 and is 200 pixels wide by 240 high. The resize is 320 wide by 240 high; this changes the original aspect ratio and makes the subject wider relative to its height. Clockwise rotation moves the subject orientation by 90 degrees; the 512-square dimensions stay the same.

Cropping selects a region without rescaling its pixels, resizing resamples the grid, and rotation rearranges locations. These operations support inspection, document processing and preparation of inputs for recognition models. The saved comparison and manifest document every operation without a hard-copy report.
