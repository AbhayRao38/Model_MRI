# Model Card — MRI MCI Classifier

Model details
- Model type: VGG16 / ensemble K-fold
- Framework: TensorFlow
- Input: preprocessed 128x128x3 MRI slices (grayscale converted to 3 channels)
- Output: probability over classes [healthy, mci]

Performance
- Report performance metrics in `reports/` and include validation folds information.

Intended use
- Clinical screening support only. Not a standalone diagnostic. Use with clinician oversight.

Security & Privacy
- Do not include patient PHI in the repository. Store datasets separately and reference via config.
