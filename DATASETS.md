# Dataset policy

This repository contains dataset integration code, not datasets. Users must obtain every dataset independently and comply with its current licence, access agreement, consent terms, and institutional requirements.

## Referenced data

- MUG facial-expression paths appear in the legacy split and training code.
- Celeb and LAPIS names appear in legacy notebooks and scripts, but their exact source editions and redistribution terms have not been documented in this repository.
- Human-participant study material is not required or authorized for publication here.

No redistribution permission has been established for source images or videos, identities, landmarks, displacement maps, generated identity-bearing outputs, or extracted metadata. Consequently, none of those artifacts belongs in the public repository.

## Local layout

The tracked Compose profile requires an explicit private data location:

```bash
export TEMPORAL_COHERENCE_DATA_DIR=/absolute/path/to/private/data
```

The directory is mounted into the container at `/home/jupyter/data`. It must remain outside the repository. The `.dockerignore` excludes common data, checkpoint, notebook, media, and generated-output paths from container build contexts.

## Prohibited public artifacts

- source dataset images or videos;
- face crops, frames, animations, or generated examples tied to dataset identities;
- landmarks, displacement maps, or other biometric derivatives without documented authority;
- participant responses, demographics, free text, recruitment or consent records, identifiers, timestamps, or study metadata;
- model checkpoints or optimizer state.

Current notebooks are retained as source-only research records with outputs and execution counts cleared. Their historical output-bearing versions are not approved for public release.
