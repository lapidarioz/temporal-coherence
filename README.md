# Temporal coherence for facial video synthesis

Research prototype for image-to-video facial-expression synthesis with explicit temporal conditioning. This repository is associated with the 2025 IEEE Access article *Enhancing Temporal Coherence in Image-to-Video Facial Expression Synthesis: A Dual-Loss Framework for Smoother Generation* by Rafael Luiz Testa, Ariane Machado-Lima, and Fátima L. S. Nunes.

## Current status

The recovered generator and temporal discriminator construct successfully in the repository-evidenced TensorFlow 2.14.0-rc1 environment. This does **not** establish article-result reproduction:

- no project weights or checkpoints are published;
- datasets must be obtained independently under their own terms;
- the current split and training constants have unresolved differences from the article description;
- no article table or metric has been reproduced from the current public-ready tree;
- current notebooks have no stored outputs or face media.

See [the environment record](docs/environment.md), [legacy split audit](docs/splits.md), and [repository assessment](docs/portfolio_review/01_repository_assessment.md) for provenance and limitations.

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
- `docs/portfolio_review/`: evidence-labelled audit, backlog, and approval record

## Citation and results

The associated article may be cited by title, authors, journal, and year as given above. This repository intentionally includes no DOI, BibTeX block, article file, unapproved result summary, or reproduction claim.

## Licence status

No project-wide software licence is currently granted. The article's Creative Commons licence does not automatically apply to this code. A root `LICENSE` will be added only after contributor, institutional, and third-party compatibility review. Existing third-party file headers remain in force; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Publication warning

The cleaned working tree contains no stored notebook outputs or detected credential leak, but earlier Git commits retain large notebook-output blobs and a deleted code-server password verifier. Do not publish the existing Git history until a history-safe release decision is completed.
