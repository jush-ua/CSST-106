---
name: CSST106 Vision laboratory
description: A local educational workbench for seven computer vision experiments.
colors:
  primary: "#116D60"
  canvas: "#FFFFFF"
  secondary-background: "#F2F5F4"
  text: "#172C2A"
  sidebar-border: "#dce5e1"
  selection: "#c2e7d8"
typography:
  body:
    fontFamily: "sans serif"
  headline:
    fontSize: "2.35rem"
    fontWeight: 650
    letterSpacing: "-.025em"
  title:
    fontSize: "1.5rem"
spacing:
  content-top: "2rem"
  content-bottom: "4rem"
  heading-top: "1.25rem"
components:
  canvas:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.text}"
  sidebar:
    backgroundColor: "{colors.secondary-background}"
---

# Design System: CSST106 Vision laboratory

## Overview

**Creative North Star: "The Laboratory Workbench"**

A light, practical workspace for running educational vision experiments on a local Windows computer. A numbered lab sidebar gives the seven experiments a stable home; the main canvas keeps image inputs, parameter controls, measured outputs and written interpretation close together.

Scientific content and actual processing comparisons carry the visual interest. Streamlit native controls provide a consistent interaction language. This document records the implementation in `app.py` and `.streamlit/config.toml`; it is based on source inspection, with no browser screenshot or computed-style verification. Framework defaults are deliberately left unspecified where the app does not override them.

**Key Characteristics:**

- White working canvas, pale green-gray secondary surface and restrained teal accent.
- Readable scientific descriptions, captions, tables and explicit settings.
- Side-by-side image comparisons and downloadable experiment evidence.
- Honest empty, running, failed and data-dependent states.

## Colors

The palette combines a muted teal accent with light neutral surfaces and dark green-black text.

### Primary

- **Workbench Teal** (`primary`): Streamlit's configured primary accent, the histogram line and the custom keyboard focus outline.

### Neutral

- **White Canvas** (`canvas`): Main working background.
- **Pale Green-Gray** (`secondary-background`): Streamlit secondary background, including the sidebar treatment.
- **Deep Green Ink** (`text`): Configured interface text.
- **Soft Green Divider** (`sidebar-border`): One-pixel sidebar right border.
- **Mint Selection** (`selection`): Browser text selection background.

Scientific images, detector annotations and semantic class visualizations carry data colors. They are not additional interface palette tokens.

## Typography

The configured family is Streamlit's `sans serif` option. The project does not pin a concrete font family or override the framework's body size, line height, control label scale or monospace font.

- **Headline:** H1 uses the frontmatter headline size, weight and tight tracking for page and sidebar titles.
- **Title:** H2 uses the frontmatter title size with the established heading top margin.
- **Body and labels:** Native Streamlit text, labels, captions and expanders. Preserve scientific notation, parameter units and readable explanatory prose.
- **Numbers:** Metric values request `font-variant-numeric: tabular-nums`.
- **Code:** Native `st.code` blocks communicate exact experiment settings and worker logs.

## Layout

`st.set_page_config(layout='wide')` supplies the application shell. The content container is capped at `1320px`, with `2rem` top and `4rem` bottom padding. H2 top margin is `1.25rem`.

Sidebar radio navigation contains Overview, labs 01–07 and Saved experiments. The sidebar includes course identity, reviewer context and an external class link. The app defines no custom responsive breakpoints; sidebar collapse, control wrapping and small-screen behavior remain Streamlit responsibilities and have not been visually verified.

Image source controls use columns in a `1:2` ratio. Parameter groups commonly use two or three equal columns. The shared result renderer distributes images across at most three columns, each with a caption and download action. Two-column before/after comparisons make individual pixel changes inspectable. Annotation and region-seed interactive images are scaled to at most `850px` wide while preserving aspect ratio and mapping clicks to original coordinates. Observation text areas are `120px` high.

## Elevation & Depth

The custom CSS adds no shadows. Tonal separation between the white canvas and secondary surface, a sidebar border, native expanders and tab organization establish hierarchy. Streamlit may supply its own control depth; the app does not redefine it.

## Shapes

The app does not define a custom corner-radius scale. Keep Streamlit native button, input, tab and expander shapes. The sidebar divider is `1px solid` in the sidebar-border color. Links use `text-underline-offset: 3px`.

## Components

### Navigation

A native sidebar radio list presents numbered lab names as full descriptive labels. Overview and Saved experiments bracket the lab sequence. Preserve the native selected and keyboard states; no custom selected-row styling is defined.

### Inputs and actions

Use native selectboxes, sliders, number fields, checkboxes, uploaders and text areas. Forms batch related parameter changes where implemented. The main experiment action uses `type='primary'`; downloads, saving and auxiliary actions retain native secondary behavior. The app's custom focus rule gives `button:focus-visible` and `input:focus-visible` a `3px solid` teal outline with `2px` offset. Other control states remain framework defaults.

### Result comparisons

Show actual processed images with meaningful method captions, per-image PNG downloads, optional collapsed metrics JSON, a Save experiment action and a Settings used expander. Preserve input provenance captions and invalidate old results when the input source changes. Tables and histograms complement images rather than replacing them.

### Scientific interpretation

An expanded Observations and interpretation expander pairs a lab-specific prompt with a persistent text area and Save observations action. PCA includes measured tables, eigenface and prediction images, plus its digital PDF download and editable visual interpretation.

### Long-running work

Native tabs separate YOLO inference, personal dataset preparation and training; segmentation separates classical methods and semantic classes. Job panels expose the selected experiment, state, progress when available, cancellation and an expandable worker log. Use native success, warning, info and error messages for actual status, including missing data or unavailable capabilities.

## Do's and Don'ts

### Do:

- **Do** keep inputs, controls, comparable outputs and interpretation together.
- **Do** retain the configured teal and neutral palette and native Streamlit interaction patterns.
- **Do** label public samples, personal data, settings and measured results explicitly.
- **Do** preserve visible keyboard focus and image captions.
- **Do** retain digital result exports, including the PCA PDF.

### Don't:

- **Don't** present decorative or synthetic images as measured experiment output or personal training photos.
- **Don't** report unexecuted training, unavailable SURF output or unmeasured scores as complete.
- **Don't** invent typography, radius, shadow or responsive tokens absent from the implementation.
- **Don't** treat source inspection or AppTest as screenshot-based visual QA.
