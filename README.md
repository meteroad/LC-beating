# LC-beating

This repository contains the research code for LC-beating, a low-latency beat and downbeat activation estimation framework. It includes the training pipeline, evaluation utilities, neural network definitions, and reference inference code used for the corresponding paper.

The repository is released as source code only. Training datasets, generated checkpoints, logs, and audio examples are not included. Please prepare licensed beat/downbeat datasets separately and point the code to your local data directory.

## Repository Structure

```text
.
├── networks/          # CRNN, TCN, and Transformer variants
├── param_file/        # Dataset and training configuration
├── train.py           # Training / testing entry point
├── trainer.py         # Training loop
├── simple_test.py     # Evaluation wrapper
├── eval.py            # Beat/downbeat metrics
├── inference.py       # Reference realtime inference interface
└── network_infer.py   # Frame-wise inference wrapper
```

## Environment

The code was developed with Python 3.7 and PyTorch. A minimal environment can be installed with:

```bash
pip install -r requirements.txt
```

Some legacy CRNN utilities depend on BeatNet-style log-spectrogram code. If you use those modules, set:

```bash
export BEATNET_PATH=/path/to/BeatNet
```

## Data

The training code expects HDF5 beat/downbeat datasets. Set the dataset root with:

```bash
export LC_BEATING_DATA_ROOT=/path/to/beat_chord_hdf5
```

Dataset names are configured in `param_file/params.py`.

## Training

Example command:

```bash
python train.py \
  --mode train \
  --network lc42_dt_1 \
  --mark round4_device1 \
  --eval_mode online \
  --peak_type simple
```

Checkpoints and TensorBoard logs are written to local output directories such as `model/`, `model_pp/`, `vis_scalar/`, and `vis_scalar_pp/`.

## Evaluation

Use test mode with a checkpoint number:

```bash
python train.py \
  --mode test \
  --network lc42_dt_1 \
  --mark round4_device1 \
  --check_num 135 \
  --eval_mode online
```

## Inference

`inference.py` provides a reference realtime interface. The public repository does not include pretrained checkpoints or demo audio; load your own trained checkpoint and licensed audio files.

## Citation

If you use this code, please cite:

Xinlu Liu, Jiale Qian, Qiqi He, Yi Yu, and Wei Li. "LC-Beating: An Online System for Beat and Downbeat Tracking using Latency-Controlled Mechanism." In *2023 IEEE International Conference on Multimedia and Expo (ICME)*, pp. 1098-1103, 2023. [DOI: 10.1109/ICME55011.2023.00192](https://doi.org/10.1109/ICME55011.2023.00192).

```bibtex
@inproceedings{liu2023lcbeating,
  author    = {Xinlu Liu and Jiale Qian and Qiqi He and Yi Yu and Wei Li},
  title     = {{LC-Beating}: An Online System for Beat and Downbeat Tracking using Latency-Controlled Mechanism},
  booktitle = {2023 IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2023},
  pages     = {1098--1103},
  publisher = {IEEE},
  doi       = {10.1109/ICME55011.2023.00192},
  url       = {https://doi.org/10.1109/ICME55011.2023.00192}
}
```

## License

This source code is released under the MIT License. See `LICENSE`.
