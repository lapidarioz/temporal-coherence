# Ethics, privacy, and biometric-data governance

This project processes facial imagery and is associated with an article that reports a human-participant evaluation. That context does not authorize publication of participant records or identity-bearing dataset artifacts.

## Public-artifact boundary

The public-ready tree may contain research source code, dataset-free tests, configuration documentation, and a citation to the associated article. It must not contain:

- participant-level or anonymized responses;
- demographics, free-text responses, recruitment or consent records;
- survey exports, identifiers, IP addresses, timestamps, or study metadata;
- face images, videos, landmarks, displacement maps, or generated identity-bearing outputs without separately documented publication authority;
- model weights trained on facial datasets.

No participant-study file was identified by filename in the current tree or retained file-name history audit. This is not proof that historical notebook images are safe: the output-bearing notebook blobs remain prohibited because their source identities and redistribution rights are unresolved.

## Use expectations

- Obtain documented authority for every input image or video.
- Respect dataset terms, consent restrictions, purpose limitation, and deletion requirements.
- Minimize retained data and keep it outside the repository and container build context.
- Do not infer identity, sensitive traits, health, emotion truth, or demographic characteristics from generated or source faces.
- Do not use this prototype to deceive, impersonate, surveil, harass, or make high-stakes decisions.

## Study claims

The repository does not reproduce participant-study results and does not publish participant analysis data. Any article-reported statement remains distinct from a newly reproduced result and requires its own approved evidence.

## Historical publication risk

Clearing notebook outputs in the current tree does not remove earlier facial imagery from Git objects. Under the current no-history-rewrite decision, the existing repository history must remain private. A public release needs a history-safe strategy that excludes prohibited blobs while preserving any required private archive.
