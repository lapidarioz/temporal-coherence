# Third-party notices and unresolved provenance

This inventory records source evidence visible in the repository. It is not a complete dependency licence report and does not grant a project-wide software licence.

## Source-derived files

| Repository file | Header/source evidence | Stated licence | Status |
|---|---|---|---|
| `fid.py` | Adapted from the [TensorFlow GAN CIFAR utility](https://github.com/tensorflow/gan/blob/3a80f96fa1c9a424d13db9b139af9677e4a8982c/tensorflow_gan/examples/cifar/util.py); copyright The TensorFlow GAN Authors | Apache License 2.0 | Header retained; modification/provenance review still required |
| `frechet_video_distance.py` | Retrieved from the [Google Research FVD implementation](https://github.com/google-research/google-research/tree/master/frechet_video_distance); copyright The Google Research Authors | Apache License 2.0 | Header retained; modification/provenance review still required |
| `film_losses.py` | Copyright Google LLC; consistent with the official [FILM frame-interpolation repository](https://github.com/google-research/frame-interpolation) | Apache License 2.0 | Header retained; exact upstream file/revision still requires confirmation |
| `vgg19_loss.py` | Copyright Google LLC; VGG feature-loss implementation associated with the official [FILM repository](https://github.com/google-research/frame-interpolation) | Apache License 2.0 | Header retained; exact upstream file/revision still requires confirmation |

The broken `PWCNet` gitlink and obsolete optical-flow loss copy were removed from the current tree in commit `dc4dc4e0c46a47c2849db1e284102f0bbb8c759c`. Their historical presence and provenance remain visible in prior commits.

## Package dependencies

`requirements/research.lock` pins a legacy Python environment but does not reproduce package licence texts or establish compatibility. Before a project licence is added, review the licences and notices of every distributed dependency and container base, including TensorFlow and its companion packages, DLib/face-recognition, MediaPipe, OpenCV, Matplotlib, SciPy, pandas, ImageIO, PyLops, and PyWavelets.

## Project licence status

Apache-2.0 is the owner's selected future project licence, but contributor, institutional, and third-party compatibility checks remain incomplete. There is intentionally no root `LICENSE` yet.
