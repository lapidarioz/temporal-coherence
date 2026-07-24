# Enhancing temporal coherence in facial video synthesis

Research code accompanying the article:

> Rafael Luiz Testa, Ariane Machado-Lima, and Fátima L. S. Nunes, “Enhancing Temporal Coherence in Image-to-Video Facial Expression Synthesis: A Dual-Loss Framework for Smoother Generation,” *IEEE Access*, vol. 13, pp. 170876–170894, 2025. [doi:10.1109/ACCESS.2025.3612820](https://doi.org/10.1109/ACCESS.2025.3612820)

The project investigates image-to-video facial-expression synthesis with explicit temporal conditioning. The generator uses the preceding synthesized frame, together with source and landmark-derived inputs, while a temporal discriminator evaluates consecutive frames and their difference.

## Current status

The recovered generator and temporal discriminator construct successfully in the repository-evidenced TensorFlow 2.14.0-rc1 environment. This does **not** establish article-result reproduction:

- no project weights or checkpoints are published;
- datasets must be obtained independently under their own terms;
- the current split and training constants have unresolved differences from the article description;
- no article table or metric has been reproduced from the current public-ready tree;
- current notebooks have no stored outputs or face media.

## Architecture validation

The dataset-free CPU smoke image is pinned by immutable digest and does not download data or weights:

```bash
docker build --file Dockerfile.cpu --tag temporal-coherence-smoke:p0-09 .
docker run --rm temporal-coherence-smoke:p0-09
```

It validates the recovered four-input generator and three-input temporal discriminator. The host-only hygiene suite can be run with Python 3.11:

```bash
python3.11 -m unittest discover -s tests -v
```

TensorFlow construction tests are skipped outside the pinned image when TensorFlow is unavailable.

## Legacy research environment

The direct research inputs are in `requirements/research.in`; the reviewed Python 3.11/Linux lock is in `requirements/research.lock`. The GPU/Jupyter profile remains a legacy development environment and has not been runtime-validated on this host.

To build it, point Compose at a private, locally obtained dataset directory:

```bash
export TEMPORAL_COHERENCE_DATA_DIR=/absolute/path/to/private/data
./configure_docker.sh
./run.sh
```

Jupyter is published on `127.0.0.1:5555` only. Do not expose it to an untrusted network.

## Data and responsible use

No face examples, participant-study material, dataset files, derived biometric artifacts, or model weights are authorized for publication here. Review [DATASETS.md](DATASETS.md), [ETHICS.md](ETHICS.md), [MODEL_CARD.md](MODEL_CARD.md), and [SECURITY.md](SECURITY.md) before using the research scripts.

The code manipulates facial imagery and relies on face landmarks. Use only data for which you have appropriate access, consent, and processing authority. Do not use this prototype for identity inference, surveillance, impersonation, high-stakes decisions, or claims about demographic robustness.

## Repository map

- `model.py`: recovered generator and temporal discriminator definitions
- `loss.py`: legacy pixel, temporal, landmark, adversarial, and feature-loss code
- `landmarks.py`, `transformation.py`: landmark extraction and piecewise affine warping
- `data_generator.py`: legacy preprocessing and sequence generation
- `mug_batch.py`, `mug_vanilla.py`: legacy research training scripts
- `tests/`: dataset-free architecture and public-artifact hygiene checks

## Citation

If this code supports your research, cite the associated article:

```bibtex
@article{testa2025enhancing,
  author  = {Testa, Rafael Luiz and Machado-Lima, Ariane and Nunes, Fátima L. S.},
  title   = {Enhancing Temporal Coherence in Image-to-Video Facial Expression Synthesis: A Dual-Loss Framework for Smoother Generation},
  journal = {IEEE Access},
  year    = {2025},
  volume  = {13},
  pages   = {170876--170894},
  doi     = {10.1109/ACCESS.2025.3612820}
}
```

The article reports the complete experimental study and results. The current repository does not include the datasets, trained weights, participant material, or a verified end-to-end reproduction of those results.

## Licence

Except where otherwise noted, project-authored material is licensed under the [Apache License 2.0](LICENSE). The article's Creative Commons licence is separate and does not govern this repository. Third-party components remain subject to their original licences and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Release status

The current tree contains no datasets, face media, participant material, model weights, checkpoints, or stored notebook outputs. However, the reachable `main` history still contains the previously identified output-bearing notebook blobs and deleted code-server password verifier, so this history is not yet suitable for public release. This history status is separate from the software-licensing and scientific-reproduction limitations above.
