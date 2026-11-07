# evans-blue-mapper

Contour-based template registration and semi-automated detection of Evans blue extravasation spots in photographs of rat forelimb skin.

Photographs of Evans blue-stained limbs differ in viewing angle, scale, orientation and cropping, and limb shape differs between animals. This software maps every photograph onto a common, view-specific anatomical template so that spots from many animals can be compared and summarized quantitatively:

1. **Landmarks**: the distal tip, the wrist (narrowest point) and the proximal cut edge are detected automatically from the tissue outline, after pose normalization.
2. **Correspondence**: the outline is divided into five landmark-delimited segments and resampled to 400 corresponding points (arc-length or shape-based dynamic time warping).
3. **Template**: a view-specific template is built by generalized Procrustes analysis and can be edited in a GUI.
4. **Registration**: each photograph is warped to the template by a thin-plate spline fitted to the outline correspondences.
5. **Detection**: spots are detected in template coordinates with an adaptive blue-index threshold (median + k·MAD), with optional click-based manual correction.
6. **Integration**: spot coordinates are exported in template, relative and grid-cell form, and groups are summarized as spot-frequency maps.

If you use this software, please cite the article (see [Citation](#citation)).

## Installation

Python 3.10 or later (tested with Python 3.12 on Windows 11).

```
git clone https://github.com/ipcng00/evans-blue-mapper.git
cd evans-blue-mapper
pip install -r requirements.txt
```

The graphical user interfaces use tkinter, which is included in the standard Python installers for Windows and macOS (on Linux, install the `python3-tk` package).

## Input data

The photographs used in the article are **not included** and cannot be made publicly available. Use your own photographs, organized as follows:

```
mapping data/
├── templates/                 # view-specific templates (included)
├── C5/                        # one folder per group; name must start with "C"
│   ├── medial/                # one folder per view; the folder name must contain
│   ├── lateral/               #   medial, lateral, dorsal or palmar
│   ├── dorsal/                #   (the Korean keywords 내측, 외측, 손등, 손바닥 also work)
│   └── palmar/
└── C6/ ...
```

- Each image must be a PNG with a **transparent background** (alpha channel); the opaque pixels define the tissue outline. Background removal can be done in any image editor.
- All limbs are expected to be photographed from the same side (e.g. left forelimbs). Photographs of a contralateral or mirrored limb can be flipped in the registration GUI or with `--flip`.

## Usage

### Graphical user interfaces

| Program | Purpose |
|---|---|
| `python limb_template_gui.py` | Build a template from selected photographs and edit its outline (drag, add or delete control points). |
| `python limb_mapping_gui.py` | Register a folder of photographs, adjust the spot threshold with a slider, correct spots by clicking (left click adds, right click removes) and save spot tables and frequency maps. |
| `python validation_annotate_gui.py` | Blinded manual annotation of anatomical landmarks and spot centers on the original photographs, for validation. |

The interfaces are available in English and Korean: choose **Language** in the menu bar, or start with `--lang en` / `--lang ko`. Text inside plots and saved figures is always in English.

### Command line

```
# build templates for all views (DTW correspondence is recommended for the palmar view)
python limb_mapping.py template --view medial lateral dorsal
python limb_mapping.py template --view palmar --dtw

# register all photographs of one folder and detect spots with the adaptive threshold
python limb_mapping.py register --folder "mapping data/C6/palmar" --view palmar --k-mad 4
```

### Output

Results are written to `<view folder>_registered/` next to the input folder:

| File | Content |
|---|---|
| `*_registered.png` | Photograph warped to template coordinates |
| `*_bluemask.png`, `*_bluedots.png` | Detected spot mask and overview |
| `blue_dots.csv` | One row per spot: template coordinates (`x_tpl`, `y_tpl`), coordinates relative to the limb axis (`x_rel`, `y_rel`), grid cell (column letter and row number of the 50-pixel grid), original-image coordinates, area, mean blue index and source (automatic or manual) |
| `summary.csv` | One row per photograph: registration Dice, threshold, number of spots |
| `_blue_frequency_map.png`, `blue_frequency.npy` | Fraction of animals with a spot at each template location |

## Validation and reproduction of the article

| Script | Purpose |
|---|---|
| `validation_eval.py` | Evaluates manual annotations: landmark target registration error for each registration model, inter-rater agreement, and precision, recall, F1 and ICC of automatic spot detection. |
| `limb_validation.py` | Comparison registration models (axis, similarity, affine, TPS) and evaluation metrics. |
| `paper_experiments.py` | Computational experiments of the article (heterogeneity, registration comparison, simulated deformations, pose/resolution/brightness robustness, synthetic spots, parameter sensitivity). |
| `paper_figures.py` | Figures of the article from the experiment results. |

```
python paper_experiments.py all --out results
python paper_figures.py --root . all
```

These scripts require the original photographs and are provided to document the analyses.

## Repository contents

```
limb_mapping.py              core library and command-line interface
limb_i18n.py                 interface language (English / Korean)
limb_template_gui.py         template construction GUI
limb_mapping_gui.py          registration, detection and correction GUI
validation_annotate_gui.py   blinded annotation GUI
validation_landmarks.json    landmark list for validation
validation_eval.py           validation analysis
limb_validation.py           comparison models and metrics
paper_experiments.py         experiments of the article
paper_figures.py             figures of the article
mapping data/templates/      templates used in the article
```

Code comments are partly in Korean.

## Citation

Cho SJ, Kim S, Yeon S, Kang N, Choi KH, Kang SY, Kim HY, Ryu Y. Standardized mapping of Evans blue extravasation spots: contour-based template registration and semi-automated detection. *Integrative Medicine Research* (submitted).

Software DOI: [10.5281/zenodo.24026503](https://doi.org/10.5281/zenodo.24026503) (Zenodo).

## License

MIT License (see `LICENSE`).

## Contact

Yeonhee Ryu, Korea Institute of Oriental Medicine (yhryu@kiom.re.kr)
