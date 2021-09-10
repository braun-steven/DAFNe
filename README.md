# DAFNe: A One-Stage Anchor-Free Deep Model for Oriented Object Detection

## Docker Setup

Use the `Dockerfile` to build the necessary docker image:

``` bash
docker build -t dafne .
```

## Training

Check out `./configs/` for different pre-defined configurations for the DOTA 1.0, DOTA 1.5 and HRSC2016 datasets. Use these paths as argument for the `--config-file` option below.


### With Docker

Use the `./tools/run.py` helper to start running experiments

``` bash
./tools/run.py --gpus 0,1,2,3 --config-file ./configs/dota-1.0/1024.yaml
```

### Without Docker

``` bash
NVIDIA_VISIBLE_DEVICES=0,1,2,3 ./tools/plain_train_net.py --num-gpus 4 --config-file ./configs/dota-1.0/1024.yaml
```

## Pre-Trained Weights

| Dataset  | mAP   | Config                                                          | Weights                                    |
|----------|-------|-----------------------------------------------------------------|--------------------------------------------|
| HRSC2016 | 87.76 | [hrsc_r101_ms](./configs/pre-trained/hrsc_r101_ms.yaml)         | [hrsc-ms.pth](weights/hrsc-ms.pth)         |
| DOTA 1.0 | 76.95 | [dota-1.0_r101_ms](./configs/pre-trained/dota-1.0_r101_ms.yaml) | [dota-1.0-ms.pth](weights/dota-1.0_ms.pth) |
| DOTA 1.5 | 71.99 | [dota-1.5_r101_ms](./configs/pre-trained/dota-1.5_r101_ms.yaml) | [dota-1.5-ms.pth](weights/dota-1.5_ms.pth) |


### Pre-Trained Weights Usage

``` bash
./tools/run.py --gpus 0 --config-file <CONFIG_PATH> --opts "MODEL.WEIGHTS <WEIGHTS_PATH>"
```
