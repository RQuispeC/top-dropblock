Top Batch DropBlock for Person Re-Identification
===========

## Installation

Make sure your [conda](https://www.anaconda.com/distribution/) is installed.


```bash

    # cd to your preferred directory and clone this repo
    git clone https://github.com/KaiyangZhou/deep-person-reid.git

    # create environment
    cd deep-person-reid/
    conda create --name torchreid python=3.7
    conda activate torchreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop
```

## Train and Test

We made available training and testing scripts inside `configs`, update `configs/im_top_bdnet_train.yaml` accordingly to the dataset you are using:

For Market1501
```yaml
data:
    sources: ['market1501']
    targets: ['market1501']
```
For DukeMTMC-ReID
```yaml
data:
    sources: ['dukemtmcreid']
    targets: ['dukemtmcreid']
```
For CUHK03 Labeled version
```yaml
data:
    sources: ['cuhk03']
    targets: ['cuhk03']
cuhk03:
    labeled_images: True
```
For CUHK03 Detected version
```yaml
data:
    sources: ['cuhk03']
    targets: ['cuhk03']
cuhk03:
    labeled_images: False
```

To train Top-BDBnet, do

```bash

python scripts/main.py \
--config-file configs/im_top_bdnet_train.yaml \
--root $PATH_TO_DATA \
--gpu-devices 0
```

To test Top-BDBnet, download the provided trained models and update `configs/im_top_bdnet_test.yaml` with the dataset and path to saved model:

```yaml
model:
    load_weights: $PATH_TO_MODEL

test:
    rerank: False # Update this if you want to use re-ranking
    visrank: False # Update this if you want to visualize activation maps
```

Then do

```bash
python scripts/main.py \
--config-file configs/im_top_bdnet_test.yaml \
--root $PATH_TO_DATA \
--gpu-devices 0
```

Results
--------

| Dataset       | mAP  | Rank-1 | mAP (RK)| Rank-1 (RK)  | Models|
| ------------- |:----:|:------:|:-------:|:------------:|:-----:|
| Market1501    | 86.7 | 95.3   | 94.4    | 95.8         |[link](#)|
| DukeMTMC-ReID | 75.1 | 87.8   | 88.7    | 90.4         |[link](#)|
| CUHK03(L)     | 75.6 | 79.1   | 88.2    | 86.6         |[link](#)|
| CUHK03(D)     | 72.8 | 77.0   | 86.0    | 84.6         |[link](#)|


Citation
---------
If you find this code useful to your research, please cite the following publication.

```bash
@article{quispe2020topbdnet,
  title={Top Batch DropBlock for Person Re-Identification},
  author={Rodolfo Quispe and Helio Pedrini},
  journal={},
  year={2020}
}
```

This repo is based on [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid), for further questions regarding data setup and others take a look to their [documentation](https://kaiyangzhou.github.io/deep-person-reid/).