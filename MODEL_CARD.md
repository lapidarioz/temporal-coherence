# Model card: temporal-coherence research prototype

## Model status

No trained model, checkpoint, or optimizer state is distributed. This card describes the constructible architecture in `model.py`, not a validated released model.

## Architecture

The generator is a U-Net-style image generator with four inputs:

- a neutral source frame;
- the previous generated or target frame;
- a warped reference frame;
- a landmark-distance or displacement representation.

For the tested 256×256 RGB configuration, it emits one 256×256 RGB frame. The temporal discriminator receives the previous frame, current frame, and their difference as a nine-channel concatenation.

## Intended use

- research into temporally conditioned facial-expression video synthesis;
- dataset-free architecture inspection and regression testing;
- controlled reproduction work after the paper configuration, split, data rights, and checkpoint provenance are resolved.

## Out-of-scope use

- impersonation, deception, harassment, surveillance, or identity inference;
- clinical, employment, education, policing, or other high-stakes decisions;
- public upload services or processing data without appropriate consent and authority;
- claims of real-time performance, production readiness, demographic fairness, in-the-wild generalization, or article-result reproduction.

## Inputs, outputs, and data

Legacy research code assumes aligned facial imagery and landmark-based preprocessing. Dataset access is not provided. No public example media is authorized. See `DATASETS.md`.

## Evaluation status

The generator and discriminator shapes pass dataset-free construction tests under TensorFlow 2.14.0-rc1. Training behavior, checkpoints, FID/FVD, temporal metrics, human-study results, subject-disjoint evaluation, and article tables have not been reproduced from the cleaned tree.

## Known limitations

- subject `084` appears in both legacy main train and test lists;
- current training constants differ from the article description;
- face-detection failure can trigger a legacy synthetic-landmark fallback;
- eye-centred alignment described by the article has not been located in the current implementation;
- the dependency stack is legacy and the GPU profile is not runtime-validated here;
- no robustness, demographic, misuse, privacy-leakage, or memorization evaluation is available.
