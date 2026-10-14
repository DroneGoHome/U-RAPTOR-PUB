# Data and Model License

Copyright (c) 2025-2026 DroneGoHome

The CageDroneRF (CDRF) dataset and related materials are licensed under the
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License (CC BY-NC-SA 4.0)**.

Full legal text: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>

## What this license covers

- The CageDroneRF dataset, however obtained (including through the
  [AeroDefense Data Request portal](https://aerodefense.tech/u-raptor-data-request)):
  raw RF recordings, processed samples, spectrograms, and annotations.
- Dataset metadata files in this repository (for example `temp_meta.json`,
  `src/yolo/temp_meta.json`, `src/yolo/temp_test_meta.json`).
- Trained model weights and checkpoints released by this project, except
  YOLO checkpoints (see below).

Source code is **not** covered by this file. See `LICENSE` and the "License"
section of `README.md`.

## Summary (not a substitute for the legal text)

You may share and adapt the dataset and the models this file covers, provided that:

- **Attribution**: you credit the authors by citing the CageDroneRF paper
  (see `README.md`), link to this license, and indicate any changes. Citing
  the paper is the manner of attribution the licensor requests under
  Section 3(a) of the license.
- **NonCommercial**: you do not use the material for commercial purposes.
  This includes selling it, using it to build or train commercial products
  or services, and using it in paid work for others.
- **ShareAlike**: if you share adapted material (for example modified,
  filtered, or re-annotated versions of the dataset), you share it under
  this same license.

Trained models released by this project are licensed under CC BY-NC-SA 4.0
directly. Whether a model that someone else trains on the dataset is itself
"adapted material" depends on the jurisdiction, so this file does not claim
that such models automatically inherit this license. Training such a model
for a commercial purpose is still a commercial use of the dataset.

## YOLO checkpoints

Checkpoints produced by the YOLO pipeline in `src/yolo/` are fine-tuned from
Ultralytics YOLO models. Ultralytics states that trained and fine-tuned YOLO
models are licensed under AGPL-3.0 by default
(<https://www.ultralytics.com/license>). Those checkpoints are therefore
released under AGPL-3.0, not CC BY-NC-SA 4.0. The dataset used to train them
remains under CC BY-NC-SA 4.0.

## Commercial use

Commercial use of the dataset or of the models this file covers requires a
separate written license. YOLO checkpoints are covered by AGPL-3.0 instead.
Contact AeroDefense via <https://aerodefense.tech>.

## Third-party data

Some tooling in this repository downloads third-party datasets (for example
`src/data/rfuav.py` downloads RFUAV from Hugging Face). Those datasets are
governed by their own licenses, not by this file.
