"""Reproduce public-sample evidence for Labs 01-03 and 07.

Run with the main .venv. Uses cached DeepLab weights and the prepared VOC subset
for Lab 07; run fetch_voc_subset.py first on a fresh clone. Controlled fixtures
have known masks and are explicitly distinct from personal YOLO training data.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from core import ROOT, DATA, sample, gray, manipulate, edges, contrast, segmentation, mask_scores, save_run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np

EXAMPLES = ROOT / 'examples'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8', newline='\n')


def export_case(folder, lab, params, images, metrics=None, arrays=None):
    """Keep the normal run manifest, including versions and exact image hashes."""
    run = save_run(lab, params, images, metrics, arrays)
    folder.mkdir(parents=True, exist_ok=True)
    for source in run.iterdir():
        if source.is_file():
            shutil.copyfile(source, folder / source.name)
    return folder


def gallery(folder, images, columns=3, filename='comparison.png', legend=None):
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (name, image) in zip(axes.ravel(), images.items()):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image,
                  cmap='gray', vmin=0, vmax=255)
        ax.set_title(name.replace('_', ' '), fontsize=11)
    if legend:
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(facecolor=colour, label=name) for name, colour in legend.items()],
                   loc='lower center', ncol=len(legend), frameon=False)
    fig.tight_layout(rect=(0, .06 if legend else 0, 1, 1))
    fig.savefig(folder / filename, dpi=120, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)


def seal(folder):
    path = folder / 'manifest.json'
    # Match Git's enforced LF checkout so evidence hashes survive cloning on Windows.
    for artifact in folder.rglob('*'):
        if artifact.is_file() and artifact.suffix in ('.json', '.md', '.csv'):
            artifact.write_bytes(artifact.read_bytes().replace(b'\r\n', b'\n'))
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['artifacts_sha256'] = {
        str(p.relative_to(folder)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob('*')) if p.is_file() and p != path
    }
    write_json(path, manifest)


def lab01():
    image = sample('Astronaut')
    params = dict(row=256, col=256, bgr=(0, 0, 255), roi=(64, 48, 200, 240), size=(320, 240), clockwise=True)
    images = manipulate(image, **params)
    metrics = {'input_shape_hwc': list(image.shape), 'pixel_before_bgr': image[256, 256].tolist(),
               'pixel_after_bgr': images['edited'][256, 256].tolist(),
               'pixels_with_unequal_channels': int(np.count_nonzero(np.ptp(image, axis=2))),
               'output_shapes': {k: list(v.shape) for k, v in images.items()}}
    folder = export_case(EXAMPLES / 'lab01', 'lab01', dict(source='scikit-image Astronaut (RGB converted to BGR)', **params), images, metrics)
    zoom = cv2.resize(images['edited'][251:262, 251:262], (220, 220), interpolation=cv2.INTER_NEAREST)
    gallery(folder, {'Original colour': image, 'Grayscale (one channel)': images['grayscale'],
                     'Edited pixel (enlarged)': zoom, 'Crop': images['crop'],
                     'Resize to 320 x 240': images['resized'], '90 degrees clockwise': images['rotated']})
    (folder / 'ANALYSIS.md').write_text(
        '# Lab 01 Colour image operations\n\n'
        f"The public Astronaut input is {image.shape[0]} pixels high by {image.shape[1]} wide with three BGR channels. "
        f"{metrics['pixels_with_unequal_channels']:,} pixels have unequal channel values, so this is a genuine colour demonstration. "
        'Each uint8 channel stores a value from 0 to 255. OpenCV orders colour channels blue, green, red; '
        'Matplotlib display converts back to RGB. Grayscale is a two-dimensional intensity array and discards colour distinctions.\n\n'
        f"At row 256, column 256, the BGR value changes from {metrics['pixel_before_bgr']} to [0, 0, 255] (red). "
        'Only the edited copy changes. The enlarged nearest-neighbour patch makes this one-pixel operation visible. '
        'The crop begins at x=64, y=48 and is 200 pixels wide by 240 high. The resize is 320 wide by 240 high; '
        'this changes the original aspect ratio and makes the subject wider relative to its height. Clockwise rotation moves the subject '
        'orientation by 90 degrees; the 512-square dimensions stay the same.\n\n'
        'Cropping selects a region without rescaling its pixels, resizing resamples the grid, and rotation rearranges locations. '
        'These operations support inspection, document processing and preparation of inputs for recognition models. '
        'The saved comparison and manifest document every operation without a hard-copy report.\n', encoding='utf-8')
    seal(folder)


def lab02(image_path=None):
    from lab_inputs import package_edge_input, select_input
    package_edge_input()
    image, source = select_input(2, image_path)
    images, arrays = edges(image)
    metrics = {'canny_edge_pixels': int(np.count_nonzero(images['canny'])),
               'sobel_magnitude_max': float(arrays['magnitude'].max()),
               'sobel_display_nonzero_pixels': int(np.count_nonzero(images['sobel'])),
               'input_shape': list(image.shape)}
    folder = export_case(EXAMPLES / 'lab02', 'lab02', dict(source=source, blur=3, sigma=0, sobel_depth='CV_64F', sobel_kernel=3, low=100, high=200), images, metrics, arrays)
    gallery(folder, {name: images[name] for name in ('original', 'sobel', 'canny')})
    (folder / 'ANALYSIS.md').write_text(
        '# Lab 02 Sobel and Canny comparison\n\n'
        f'Source: {source}. A real JPEG input.jpg and its source/checksum are bundled with this lab. '
        'The bundled file is a disclosed replacement, not the unavailable instructor-provided image. '
        'Measurements are recomputed from the decoded JPEG, including its compression effects. Override it with '
        '`python complete_labs.py --labs 2 --edge-image PATH_TO_input.jpg`.\n\n'
        'Both detectors use the same grayscale input and a 3 x 3 Gaussian blur with sigma 0. '
        'Sobel uses signed float64 x/y derivatives, kernel 3, and their Euclidean magnitude. '
        'Canny uses thresholds 100 and 200. The three-panel figure shows exactly original, Sobel and Canny in that order. '
        'The blurred image is saved separately. gx.npy, gy.npy and magnitude.npy preserve numerical values; '
        'only the displayed Sobel magnitude is normalized to 0-255.\n\n'
        f"Canny marks {metrics['canny_edge_pixels']:,} pixels. The raw maximum Sobel magnitude is "
        f"{metrics['sobel_magnitude_max']:.2f}; {metrics['sobel_display_nonzero_pixels']:,} pixels remain nonzero in its normalized display. "
        'These counts are descriptive, not detector-accuracy scores. Sobel magnitude retains graded responses; '
        'strong boundaries can occupy broad bands. Canny produces a binary, thinned map. '
        + ('In the packaged Camera image, the tripod and body outline stand out while much weak ground texture is suppressed. '
           if image_path is None else 'Inspect the supplied image for retained boundaries and suppressed texture. ') +
        'Some low-contrast boundaries disappear because they do not survive thresholding and hysteresis.\n\n'
        'Sobel is useful when direction and gradient strength are needed. Canny is useful for compact boundaries '
        'and contour extraction. Smoothing reduces noisy gradients but also removes fine detail; threshold choice '
        'trades missed weak boundaries against extra responses. Neither edge map assigns object identities or fills object interiors.\n', encoding='utf-8')
    seal(folder)


def channel_metrics(channels):
    rows = {}
    for name, a in channels.items():
        counts = np.bincount(a.ravel(), minlength=256)
        p = counts[counts > 0] / a.size
        rows[name] = dict(min=int(a.min()), max=int(a.max()), standard_deviation=float(a.std()),
                          percentile_95_minus_5=float(np.percentile(a, 95)-np.percentile(a, 5)),
                          entropy_bits=float(-(p * np.log2(p)).sum()))
    return rows


def lab03():
    camera = gray(sample('Camera'))
    low = np.rint(100 + camera.astype(float) * 55 / 255).astype(np.uint8)
    ramp = np.linspace(.30, 1., camera.shape[1])[None, :]
    noise = np.random.default_rng(42).normal(0, 3, camera.shape)
    uneven = np.clip(camera.astype(float) * ramp + noise, 0, 255).astype(np.uint8)
    cases = [('low_contrast', cv2.cvtColor(low, cv2.COLOR_GRAY2BGR), False, 'Camera mapped from 0-255 to 100-155'),
             ('uneven_illumination', cv2.cvtColor(uneven, cv2.COLOR_GRAY2BGR), False, 'Camera times horizontal 0.30-1.00 illumination ramp plus Gaussian noise sigma=3, seed=42'),
             ('colour_luminance', sample('Coffee'), True, 'Public scikit-image Coffee; process Y in YUV and preserve U/V before RGB conversion')]
    records = {}
    for name, image, colour, description in cases:
        images, channels = contrast(image, colour=colour)
        metrics = channel_metrics(channels);records[name] = metrics
        folder = export_case(EXAMPLES / 'lab03' / name, 'lab03', dict(source=description, clip=2., tiles=8, colour=colour), images, metrics)
        gallery(folder, images, columns=4)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (label, a) in zip(axes, channels.items()):
            ax.plot(np.arange(256), np.bincount(a.ravel(), minlength=256), color='#116d60')
            ax.set(title=label, xlabel='Intensity (Y for colour case)', ylabel='Pixel count', xlim=(0, 255))
        fig.tight_layout();fig.savefig(folder / 'histograms.png', dpi=120);plt.close(fig)
        with (folder / 'histograms.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream);writer.writerow(['intensity'] + list(channels))
            hist = [np.bincount(a.ravel(), minlength=256) for a in channels.values()]
            for i in range(256):writer.writerow([i] + [int(h[i]) for h in hist])
        seal(folder)
    folder = EXAMPLES / 'lab03'
    analysis = [
        '# Lab 03 Histogram equalization and contrast', '',
        'Three reproducible public-sample cases cover restricted dynamic range, uneven illumination with noise, '
        'and colour luminance. Each case includes original/stretch/global equalization/CLAHE images, '
        'a four-panel comparison, histograms.png and all 256 histogram bins in histograms.csv. '
        'CLAHE uses clipLimit=2 and 8 x 8 tiles. The illumination and low-contrast images are explicitly derived fixtures.', '',
        '| Case | Version | Range | Standard deviation | P95-P5 | Entropy bits |',
        '|---|---|---|---:|---:|---:|',
    ]
    for case, rows in records.items():
        for name, m in rows.items():
            analysis.append(f"| {case} | {name} | {m['min']}-{m['max']} | {m['standard_deviation']:.2f} | {m['percentile_95_minus_5']:.2f} | {m['entropy_bits']:.2f} |")
    analysis += ['',
        '## Observations', '',
        'The low-contrast original occupies only 100-155. Stretching maps these endpoints to 0 and 255, '
        'making the coat, tripod and background more distinct without inventing new intensity levels. '
        'Global equalization redistributes the populated bins through the cumulative histogram, '
        'while CLAHE changes local neighbourhood contrast. In the uneven-light case, global mapping '
        'cannot compensate independently for the dark left and brighter right. CLAHE can reveal local '
        'structure but also makes grain in smooth regions more noticeable. Its clipping limits this effect; '
        'it does not remove the added noise. Greater spread or entropy alone does not establish better visual quality.', '',
        'The colour case applies each mapping only to Y, then reconstructs colour with the original U/V channels. '
        'Processing B/G/R independently could distort colour relationships. Conversion and clipping can still alter '
        'reconstructed colours; luminance processing does not guarantee perfect colour preservation.', '',
        '## Questions and conclusion', '',
        'Histogram equalization remaps pixel values using their cumulative frequency, approximately distributing '
        'occupied intensities across the available output range. It helps reveal details compressed into a narrow '
        'brightness range, such as inspection surfaces or poorly exposed photographs. It cannot recover clipped '
        'highlights or missing information. CLAHE computes clipped local histograms and interpolates mappings '
        'between tiles, addressing local illumination differences while limiting contrast amplification. '
        'Stretching is the simpler linear endpoint mapping. Global equalization uses one nonlinear mapping '
        'for the entire image; CLAHE trades extra local computation and parameter choices for spatial adaptation. '
        'Choose by visibility of relevant structures and artifacts, rather than maximizing a histogram statistic.', '',
    ]
    (folder / 'ANALYSIS.md').write_text('\n'.join(analysis), encoding='utf-8')
    write_json(folder / 'manifest.json', {'lab': 'lab03', 'cases': [name + '/manifest.json' for name, *_ in cases], 'metrics': records})
    seal(folder)


def controlled_objects():
    """Two overlapping bright discs, known union mask, no learned labels."""
    y, x = np.indices((192, 256))
    truth = ((x-96)**2+(y-96)**2 <= 34**2) | ((x-152)**2+(y-96)**2 <= 34**2)
    clean = np.where(truth, 205, 30).astype(np.uint8)
    uneven = np.clip(clean.astype(float) * np.linspace(.25, 1.15, 256)[None, :] +
                     np.random.default_rng(42).normal(0, 4, clean.shape), 0, 255).astype(np.uint8)
    return clean, uneven, truth.astype(np.uint8)


def lab07():
    voc = DATA / 'voc_subset'
    if not (voc / 'val_0000.png').exists():
        raise FileNotFoundError('Run fetch_voc_subset.py first; Lab 07 needs the actual VOC image and truth.')
    clean, uneven, truth = controlled_objects()
    natural = cv2.imread(str(voc / 'val_0000.png'))
    semantic_truth = cv2.imread(str(voc / 'val_0000_truth.png'), cv2.IMREAD_UNCHANGED)
    binary_truth = (semantic_truth != 0).astype(np.uint8);binary_truth[semantic_truth == 255] = 255
    cases = [('touching_objects', cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR), truth, (96, 96), 15,
              'Controlled two-disc fixture: centres (x,y)=(96,96),(152,96), radius=34, foreground=205, background=30'),
             ('uneven_illumination', cv2.cvtColor(uneven, cv2.COLOR_GRAY2BGR), truth, (96, 96), 30,
              'Same disc union, illumination ramp 0.25-1.15 plus Gaussian noise sigma=4, seed=42'),
             ('texture', cv2.resize(sample('Camera'), (256, 256)), None, (128, 128), 15,
              'Public scikit-image Camera resized to 256 square; no ground truth'),
             ('natural_scene', natural, binary_truth, (natural.shape[0]//2, natural.shape[1]//2), 15,
              'VOC validation val_0000, provenance in source_manifest.json; binary truth is the union of non-background VOC classes')]
    rows = [];all_metrics = {}
    for name, image, target, seed, tolerance, description in cases:
        images, markers, metrics = segmentation(image, seed, tolerance=tolerance)
        predictions = {k: images[k] > 0 for k in ('otsu', 'adaptive', 'region', 'edge_regions')}
        predictions['watershed'] = markers > 1
        if target is not None:
            metrics['foreground_scores'] = {}
            for method, prediction in predictions.items():
                score = mask_scores(prediction.astype(np.uint8), target, 2)
                metrics['foreground_scores'][method] = score['classes'][1]
            images['ground_truth'] = np.where(target == 255, 127, target * 255).astype(np.uint8)
        metrics['foreground_pixel_counts'] = {k: int(v.sum()) for k, v in predictions.items()}
        metrics['watershed_object_regions'] = int(len(np.unique(markers[markers > 1])))
        folder = export_case(EXAMPLES / 'lab07' / name, 'lab07', dict(source=description, seed=seed, tolerance=tolerance, adaptive_block=25, adaptive_c=8, marker_fraction=.45, connectivity=4), images, metrics, {'watershed_labels': markers})
        gallery(folder, images, columns=4)
        for method in predictions:
            score = metrics.get('foreground_scores', {}).get(method, {})
            rows.append({'case': name, 'method': method, 'foreground_iou': score.get('iou'),
                         'foreground_dice': score.get('dice'), 'milliseconds': metrics['timings_ms'][method]})
        all_metrics[name] = metrics
        seal(folder)
    folder = EXAMPLES / 'lab07'
    # Change only seeds/tolerance to expose sensitivity on the very same input.
    seed_rows = [];seed_images = {}
    for name, seed, tolerance in [('object_tol8', (96, 96), 8), ('object_tol30', (96, 96), 30), ('background_tol30', (10, 10), 30)]:
        images, _, metrics = segmentation(cv2.cvtColor(uneven, cv2.COLOR_GRAY2BGR), seed, tolerance=tolerance)
        score = mask_scores((images['region'] > 0).astype(np.uint8), truth, 2)['classes'][1]
        seed_rows.append(dict(case=name, seed=seed, tolerance=tolerance, foreground_pixels=int(np.count_nonzero(images['region'])), **score))
        seed_images[name] = images['region']
    gallery(folder, seed_images, filename='seed_sensitivity.png')
    marker_rows = [];marker_images = {}
    for fraction in (.35, .65, .85):
        images, labels, metrics = segmentation(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR), (96, 96), marker_fraction=fraction)
        count = len(np.unique(labels[labels > 1]));marker_rows.append(dict(marker_fraction=fraction, seed_components=metrics['watershed_seed_components'], output_regions=count))
        marker_images['marker fraction ' + str(fraction)] = images['watershed']
    gallery(folder, marker_images, filename='marker_sensitivity.png')
    # Run semantic inference on exactly the natural-scene image used above.
    semantic_folder = folder / 'natural_scene' / 'semantic'
    semantic_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(voc / 'val_0000.png', semantic_folder / 'input.png')
    shutil.copyfile(voc / 'val_0000_truth.png', semantic_folder / 'truth.png')
    shutil.copyfile(voc / 'manifest.json', folder / 'source_manifest.json')
    (semantic_folder / 'input.ready').touch()
    from worker import semantic_job
    semantic_job(semantic_folder, {})
    semantic = json.loads((semantic_folder / 'result.json').read_text())
    classes = cv2.imread(str(semantic_folder / 'classes.png'))
    labels = cv2.imread(str(semantic_folder / 'labels.png'), cv2.IMREAD_UNCHANGED)
    legend = {name: classes[labels == int(value)][0][::-1] / 255.
              for value, name in semantic['legend'].items()}
    gallery(semantic_folder, {'Natural scene': natural, 'DeepLab classes': classes, 'DeepLab overlay': cv2.imread(str(semantic_folder / 'overlay.png'))}, legend=legend)
    write_json(folder / 'sensitivity.json', {'seed_sweep': seed_rows, 'marker_sweep': marker_rows})
    with (folder / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(folder / 'manifest.json', {'lab': 'lab07', 'cases': [name + '/manifest.json' for name, *_ in cases], 'metrics': all_metrics,
                                       'sensitivity': 'sensitivity.json', 'semantic': 'natural_scene/semantic/result.json'})
    analysis = ['# Lab 07 Segmentation comparison', '',
        'All five classical methods use the same input within each case. The controlled discs and their '
        'illumination variant have exact union masks; they are test fixtures, not personal photographs. '
        'Camera provides complex texture without invented ground truth. The natural scene is a real VOC '
        'validation pair, also passed to pretrained DeepLabV3-ResNet50. The existing five-image semantic '
        'evaluation remains in examples/semantic/five_image_scores.json.', '',
        '| Case | Method | Foreground IoU | Foreground Dice | Stage ms |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        iou = 'n/a' if r['foreground_iou'] is None else f"{r['foreground_iou']:.4f}"
        dice = 'n/a' if r['foreground_dice'] is None else f"{r['foreground_dice']:.4f}"
        analysis.append(f"| {r['case']} | {r['method']} | {iou} | {dice} | {r['milliseconds']:.2f} |")
    analysis += ['',
        'Binary scores measure foreground union only, not object identity or separation. Watershed uses labels >1 '
        'as foreground and counts boundary pixels as excluded. VOC void=255 is ignored. Classical methods are '
        'not trained to distinguish VOC categories; these objectness scores must not be equated with semantic mean IoU. '
        'Stage timings exclude shared grayscale/blur preprocessing and plotting; that preprocessing and total '
        'time are recorded separately. Single CPU timings are descriptive, not a controlled benchmark.', '',
        '## Observations from the displayed cases', '',
        f"On the clean discs, Otsu foreground IoU is {all_metrics['touching_objects']['foreground_scores']['otsu']['iou']:.4f} "
        f"and seeded growing is {all_metrics['touching_objects']['foreground_scores']['region']['iou']:.4f}. "
        'Both recover the union, but neither separates the two instances. Adaptive thresholding with positive C '
        'makes much of the uniform background white and produces a dark ring around the objects, explaining its '
        'poor foreground score here. It is not automatically better than global thresholding.', '',
        f"With the illumination ramp and noise, Otsu still achieves {all_metrics['uneven_illumination']['foreground_scores']['otsu']['iou']:.4f} "
        'because foreground and background remain largely distinguishable. The region mask truncates inside the '
        'right-hand disc as brightness departs from the seed; the sensitivity figures show how increasing tolerance '
        'recovers more of the object without fully reaching its brighter side.', '',
        'On Camera, the central seed selects only a small connected patch. Bright-foreground thresholding '
        'groups sky and ground while leaving much of the dark coat black. Filled edge contours recover some '
        'tripod and face regions but also follow texture; watershed groups broad brightness areas, not the person. '
        'There is no annotated Camera mask, so no quantitative accuracy is claimed for this case.', '',
        f"In the aircraft scene, edge filling has foreground IoU {all_metrics['natural_scene']['foreground_scores']['edge_regions']['iou']:.4f}, "
        'but includes foreground equipment and background regions. The centre seed grows mainly into the sky, '
        'and bright-foreground Otsu also selects much sky. This is a polarity/appearance mismatch with the '
        'annotated aircraft, not a claim that thresholding universally fails. DeepLab uses learned context '
        'to cover the fuselage, wings and tail much more coherently, though boundaries and small predicted '
        'fragments still need inspection.', '',
        '## Seed and watershed sensitivity', '',
    ]
    for row in seed_rows:
        analysis.append(f"- {row['case']}: seed={row['seed']}, tolerance={row['tolerance']}, {row['foreground_pixels']} pixels, foreground IoU={row['iou']:.4f}.")
    for row in marker_rows:
        analysis.append(f"- Watershed fraction {row['marker_fraction']}: {row['seed_components']} initial components, {row['output_regions']} output object regions.")
    analysis += ['',
        'Region growing accepts connected pixels within tolerance of the original seed intensity. '
        'With uneven light a small tolerance can stop inside one object, while a larger tolerance expands further; '
        'a background seed selects a different region entirely. Automatic watershed depends on foreground '
        'polarity and marker placement. Connected markers can leave touching objects merged; more markers can '
        'split them, and excessive or noisy markers can over-segment a single object. Count and inspect regions '
        'as well as foreground overlap. A high binary IoU does not prove the two discs were separated.', '',
        '## Natural scene semantic result', '',
        f"DeepLab labels the same {semantic['shape'][1]} x {semantic['shape'][0]} scene in {semantic['inference_seconds']:.2f} seconds on CPU. "
        f"Its class mean IoU is {semantic['scores']['mean_iou']:.4f}, and pixel accuracy is {semantic['scores']['pixel_accuracy']:.4f}. "
        f"Predicted labels: {', '.join(semantic['legend'].values())}. "
        'The palette, class-ID mask and overlay are saved with a legend in result.json. '
        'High pixel accuracy can be dominated by background; class IoUs expose small or spurious classes. '
        'The model recognizes a fixed vocabulary and may miss objects or assign an incorrect class at boundaries.', '',
        '## Use cases limitations and conclusion', '',
        'Thresholding is fast and appropriate for high-contrast foreground/background separation. Otsu uses one '
        'global intensity split; adaptive thresholding handles local brightness changes but its block size and '
        'constant can turn texture or background into foreground. Compare the controlled clean and uneven cases '
        'with identical geometry to isolate illumination effects.', '',
        'Region methods use connectivity and intensity similarity. Seeded growing is useful for a known homogeneous '
        'region but can leak into similar background or stop at a lighting gradient. Watershed is useful for touching '
        'objects when markers represent the intended objects; its labels have no semantic meaning.', '',
        'Edges locate intensity transitions. Closing and filling contours can recover regions with clear boundaries, '
        'but noise makes extra contours and gaps can connect object and background. Camera texture demonstrates '
        'why many edge responses are not necessarily useful object borders. Compare the Canny map with the filled '
        'edge regions and watershed boundaries rather than treating every contour as an object.', '',
        'Semantic segmentation supplies learned per-pixel categories and scene context instead of brightness groups. '
        'It requires pretrained weights, tensor preprocessing, a compatible class vocabulary and substantially '
        'more memory/computation here. Thin structures, unusual appearance and boundaries can still be wrong. '
        'No semantic training is required for this activity.', '',
        'Thresholds and fixed-size local filtering scan the image; region growing can visit the entire connected '
        'image and this Python queue has noticeable overhead. Watershed adds distance, marker and flooding work. '
        'Contour extraction depends on the number and length of boundaries. DeepLab executes many convolutional '
        'layers, with runtime and memory depending on input size, model and device. The measured tables include '
        'single-run effects and must not be read as universal speed rankings. Mask alignment, polarity, seed '
        'coordinates, void labels and the difference between instance labels and class labels are key '
        'implementation challenges. The results support choosing the method by scene and intended output, '
        'rather than declaring one method best for every image.', '',
    ]
    (folder / 'ANALYSIS.md').write_text('\n'.join(analysis), encoding='utf-8')
    seal(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labs', nargs='+', type=int, choices=(1, 2, 3, 7), default=[1, 2, 3, 7])
    parser.add_argument('--edge-image', type=Path, help='Use the actual instructor input.jpg when available.')
    args = parser.parse_args()
    for lab in args.labs:
        if lab == 2:lab02(args.edge_image)
        else:{1: lab01, 3: lab03, 7: lab07}[lab]()
        print('Completed Lab', lab, flush=True)


if __name__ == '__main__':
    main()
