# Security and privacy

## Supported scope

This is an offline research prototype, not a hosted service. It has no supported public API, upload endpoint, authentication system, or multi-user deployment.

## Safe local operation

- Keep datasets and checkpoints outside the repository.
- Use the tracked Compose profile only on a trusted machine; it binds Jupyter to `127.0.0.1:5555`.
- Retain Jupyter's runtime token and do not publish its logs or access URL.
- Do not mount Git credentials, editor credentials, SSH keys, or unrelated home-directory content into the container.
- Treat external checkpoints, NumPy objects, notebooks, and model files as untrusted unless their source and loading behavior have been reviewed.

## Historical credential notice

Commit `614f46668e969eeed6d8048c9890df6f928319d8` introduced a code-server password verifier. The current tree removes that verifier and the unused code-server installation path. Deletion does not make the historical verifier safe: the source password must be treated as compromised and rotated anywhere it was reused.

The retained Git history also contains output-heavy notebooks with facial imagery. The current no-history-rewrite decision means this repository must remain private until a history-safe publication strategy is chosen.

## Reporting

Do not include credentials, participant information, private dataset samples, or face media in a public issue. Contact the repository owner privately through an agreed channel. No public security contact address is designated yet.
