# Environment

The reported experiments used Python 3.10.13, PyTorch 2.2.2+cu121,
and CUDA 12.1 in the conda-based `deepfake_lab` container.

The complete package state recorded from that container is provided in
[`container_pip_freeze.txt`](container_pip_freeze.txt).

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The released Dockerfile reconstructs a compatible CUDA 12.1 environment;
it is not a byte-identical export of the original experimental container.

The original container contained both `opencv-python` and
`opencv-python-headless` 4.13.0.90, an unsupported combination that also
conflicts with the recorded NumPy 1.26.4 under current pip resolution.
The public environment therefore uses only
`opencv-python-headless==4.11.0.86`.

The pSBI implementation and its provenance are documented in
[`datasets/README.md`](../datasets/README.md).
