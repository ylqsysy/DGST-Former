# DGST-Former

PyTorch implementation of DGST-Former for traffic flow forecasting on PEMS03, PEMS04, PEMS07 and PEMS08.

## Requirements

```bash
pip install numpy==1.26.4 torch==2.7.1
```

Python 3.12. Install a CUDA build of PyTorch for GPU training.

## Data Preparation

```bash
python download_data.py --datasets PEMS04
python prepare_data.py --datasets PEMS04
```

Data are stored under `data/`. Omit `--datasets` to prepare all four datasets.

## Training

```bash
python train.py \
  --dataset PEMS04 \
  --device auto
```

Each run writes `best.pt`, `training.csv` and `result.json` to a new directory under `outputs/`.

## Evaluation

```bash
python eval.py --checkpoint outputs/<run>/best.pt
```

## Configuration

Training settings are read at runtime from `configs/traffic_models.xlsx` and `configs/model_defaults.json`, which are not included in this repository. Place both files under `configs/` before training. The workbook supplies learning rate, optimizer, epochs, batch size, loss, dropout and early-stopping patience; the JSON supplies architecture and runtime settings.

```bash
python train.py --dataset PEMS04 --print-config
```

## License

MIT
