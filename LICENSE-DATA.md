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
- Trained model weights and checkpoints released by this project.

Source code is **not** covered by this file. See `LICENSE` and the "License"
section of `README.md`.

## Summary (not a substitute for the legal text)

You may share and adapt the dataset and models, provided that:

- **Attribution**: you credit the authors and cite the CageDroneRF paper
  (see `README.md`), link to this license, and indicate any changes.
- **NonCommercial**: you do not use the material, or anything derived from it,
  for commercial purposes. This includes selling it, using it to build or
  train commercial products or services, and using it in paid work for
  others.
- **ShareAlike**: anything you build from the material (modified datasets,
  derived datasets, models trained on it) is distributed under this same
  license.

## Commercial use

Commercial use of the dataset or models requires a separate written license.
Contact AeroDefense via <https://aerodefense.tech>.

## Third-party data

Some tooling in this repository downloads third-party datasets (for example
`src/data/rfuav.py` downloads RFUAV from Hugging Face). Those datasets are
governed by their own licenses, not by this file.
