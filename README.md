<h1 align="center"><strong>Active Neural Mapping</strong></h1>

<p align="center">
	<a href="https://zikeyan.github.io/">Zike Yan</a>, 
	Haoxiang Yang, 
	<a href="https://scholar.google.com/citations?user=LQxSSgYAAAAJ&hl=zh-CN">Hongbin Zha</a>
</p>

<div align="center">
	<a href='https://arxiv.org/abs/2308.16246'><img src='https://img.shields.io/badge/arXiv-2308.16246-b31b1b'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 	<a href='https://zikeyan.github.io/active-INR/index.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 	<a href='https://www.youtube.com/watch?v=psPvanfh7SA&feature=youtu.be'><img src='https://img.shields.io/badge/Youtube-Video-blue'></a>
</div>

## Update
- [x] Please check our updated version: [ActiveSplat: High-Fidelity Scene Reconstruction
through Active Gaussian Splatting](https://li-yuetao.github.io/ActiveSplat/). [2024.10.29]
- [x] Please check the code of our follow-up work: [Active Neural Mapping at Scale](https://github.com/kzj18/activeINR-S). Same guidance through weight perturbation but with better planner and mapping backbone to achieve complete exploration and accurate reconstruction. [2024.10.01]
- [x] We speed up the completeness evaluation through parallel computation. [2024.03.17]

## Installation

Our environment has been tested on Ubuntu 18.04 (CUDA 10.2 with RTX2080Ti) and Ubuntu 20.04(CUDA 10.2/11.3 with RTX2080Ti). Torch1.12.1 is recommended to reproduce the results.

Clone the repo and create conda environment

```shell
git clone --recurse-submodules git@github.com:ZikeYan/activeINR.git && cd activeINR

# create conda env
conda env create -f environment.yml
conda activate activeINR
```

Install pytorch by following the [instructions](https://pytorch.org/get-started/locally/). 

```shell
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

pip install -e .
```

## Preparation

#### Simulated environment

[Habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) need to be installed for simulation. We use v0.1.7 `(git checkout tags/v0.1.7)` and install the habitat-sim with the flag `--with-cuda`.

```shell
pip install -e habitat-lab
cd habitat-sim && python setup.py install --with-cuda
```

#### Data

To run the active mapping in the simulated environment, [Gibson](https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform) dataset for Habitat-sim and the [Matterport3D](https://niessner.github.io/Matterport/#download) dataset should be downloaded. The directory for the downloaded data should be specified in the config file of `activeINR/train/configs/gibson.json` via the key `root`.

#### Trained models

We adopt the [DDPPO](https://wijmans.xyz/publication/ddppo-2019/) for point-goal navigation. All pre-trained models can be found [here](https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines/habitat_baselines/rl/ddppo). The model should be placed in `activeINR/local_policy_models` and specified in the config file of `activeINR/train/configs/gibson.json` via the key `planner`.

## Run

To run Active Neural Mapping on the `Denmark` scene of Gibson dataset, run the following command.

```python
python activeINR/train/vis_exploration.py --config activeINR/train/configs/gibson.json --scene_id Denmark
```

The logs will be saved in the `./activeINR/train/logs/` folder with actions, mesh file, checkpoints of the neural map, etc.

The mesh quality and the exploration coverage can be evaluated through the following manuscript:

```python
python activeINR/eval/eval_action.py --config activeINR/train/configs/gibson.json --scene_id Denmark --file "logs/final/gibson/Denmark/results/action.txt"

python eval/eval_mesh.py
```

## TODO

The repo is still under construction, thanks for your patience.

- [ ] Running with a live camera in ROS.
- [ ] BALD implementation.  
- [ ] Loss landscape visualization.


## Acknowledgement

Our code is partially based on [iSDF](https://github.com/facebookresearch/iSDF) and [UPEN](https://github.com/ggeorgak11/UPEN). We thank the authors for making these codes publicly available.

## Citation

```
@inproceedings{Yan2023iccv,
  title={Active Neural Mapping},
  author={Yan, Zike and Yang, Haoxiang and Zha, Hongbin},
  booktitle={Intl. Conf. on Computer Vision (ICCV)},
  year={2023}
}
```

