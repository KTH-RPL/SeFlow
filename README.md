SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving
---

[![arXiv](https://img.shields.io/badge/arXiv-2407.01702-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2407.01702)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seflow-a-self-supervised-scene-flow-method-in/self-supervised-scene-flow-estimation-on-1)](https://paperswithcode.com/sota/self-supervised-scene-flow-estimation-on-1?p=seflow-a-self-supervised-scene-flow-method-in)
[![poster](https://img.shields.io/badge/ECCV24|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://hkustconnect-my.sharepoint.com/:b:/g/personal/qzhangcb_connect_ust_hk/EWyWD-tAX4xIma5U7ZQVk9cBVjsFv0Y_jAC2G7xAB-w4cg?e=c3FbMg) 
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/fQqx2IES-VI)

![](assets/docs/seflow_arch.png)

Task: __Self-Supervised__ Scene Flow Estimation in Autonomous Driving. No human-label needed. Real-time inference (15-20Hz in RTX3090).

📜 2025/02/18: Merging all scene flow code to [OpenSceneFLow codebase](https://github.com/KTH-RPL/OpenSceneFlow) for afterward code maintenance. This repo saved README, [cluster slurm files](assets/slurm), and [quick core file](lossfunc.py) in SeFlow. The old source code branch is also [available here](https://github.com/KTH-RPL/SeFlow/tree/source).

2024/11/18 16:17: Update model and demo data download link through HuggingFace, Personally I found `wget` from HuggingFace link is much faster than Zenodo.

2024/09/26 16:24: All codes already uploaded and tested. You can to try training directly by downloading (through [HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow)/[Zenodo](https://zenodo.org/records/13744999)) demo data or pretrained weight for evaluation. 

Pre-trained weights for models are available in [Zenodo](https://zenodo.org/records/13744999)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow) link. Check usage in [2. Evaluation](#2-evaluation) or [3. Visualization](#3-visualization).

## 0. Setup

**Environment**: Clone the repo and build the environment, check [detail installation](https://github.com/KTH-RPL/OpenSceneFlow/assets/README.md) for more information. [Conda](https://docs.conda.io/projects/miniconda/en/latest/)/[Mamba](https://github.com/mamba-org/mamba) is recommended.


```bash
git clone --recursive https://github.com/KTH-RPL/OpenSceneFlow.git
cd OpenSceneFlow
mamba env create -f environment.yaml
```

CUDA package (need install nvcc compiler), the compile time is around 1-5 minutes:
```bash
mamba activate opensf
# CUDA already install in python environment. I also tested others version like 11.3, 11.4, 11.7, 11.8 all works
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

Or another environment setup choice is [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment, check more information in [OpenSceneFlow/assets/README.md](https://github.com/KTH-RPL/OpenSceneFlow/assets/README.md#docker-environment).

## 1. Run & Train

Note: Prepare raw data and process train data only needed run once for the task. No need repeat the data process steps till you delete all data. We use [wandb](https://wandb.ai/) to log the training process, and you may want to change all `entity="kth-rpl"` to your own entity.

### Data Preparation

Check [dataprocess/README.md](dataprocess/README.md#argoverse-20) for downloading tips for the raw Argoverse 2 dataset. Or maybe you want to have the **mini processed dataset** to try the code quickly, We directly provide one scene inside `train` and `val`. It already converted to `.h5` format and processed with the label data. 
You can download it from [Zenodo](https://zenodo.org/records/13744999/files/demo_data.zip)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow/blob/main/demo_data.zip) and extract it to the data folder. And then you can skip following steps and directly run the [training script](#train-the-model).

```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo_data.zip
unzip demo_data.zip -p /home/kin/data/av2
```

#### Process train data

Process train data for self-supervised learning. Only training data needs this step. [Runtime: Normally need 15 hours for my desktop, 3 hours for the cluster with five available nodes parallel running.]

```bash
python process.py --data_dir /home/kin/data/av2/preprocess_v2/sensor/train --scene_range 0,701
```

### Train the model

Train SeFlow needed to specify the loss function, we set the config of our best model in the leaderboard. [Runtime: Around 11 hours in 4x A100 GPUs.]

```bash
python train.py model=deflow lr=2e-4 epochs=9 batch_size=16 loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" "model.target.num_iters=2" "model.val_monitor=val/Dynamic/Mean"
```

Or you can directly download the pre-trained weight from [Zenodo](https://zenodo.org/records/13744999/files/seflow_best.ckpt)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow/blob/main/seflow_best.zip) and skip the training step. 

### Other Benchmark Models

You can also train the supervised baseline model in our paper with the following command. [Runtime: Around 10 hours in 4x A100 GPUs.] 
```bash
python train.py model=fastflow3d lr=4e-5 epochs=20 batch_size=16 loss_fn=ff3dLoss
python train.py model=deflow lr=2e-4 epochs=20 batch_size=16 loss_fn=deflowLoss
```

> [!NOTE]  
> You may found the different settings in the paper that is all methods are enlarge learning rate to 2e-4 and decrease the epochs to 20 for faster converge and better performance. 
> However, we kept the setting on lr=2e-6 and 50 epochs in (SeFlow & DeFlow) paper experiments for the fair comparison with ZeroFlow where we directly use their provided weights. 
> We suggest afterward researchers or users to use the setting here (larger lr and smaller epoch) for faster converge and better performance.

## 2. Evaluation

You can view Wandb dashboard for the training and evaluation results or upload result to online leaderboard.

Since in training, we save all hyper-parameters and model checkpoints, the only thing you need to do is to specify the checkpoint path. Remember to set the data path correctly also.

```bash
# downloaded pre-trained weight, or train by yourself
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/seflow_best.ckpt

# it will directly prints all metric
python eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=val

# it will output the av2_submit.zip or av2_submit_v2.zip for you to submit to leaderboard
python eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=test leaderboard_version=1
python eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=test leaderboard_version=2
```

And the terminal will output the command for you to submit the result to the online leaderboard. You can follow [this section for evalai](https://github.com/KTH-RPL/DeFlow?tab=readme-ov-file#2-evaluation).

Check all detailed result files (presented in our paper Table 1) in [this discussion](https://github.com/KTH-RPL/DeFlow/discussions/2).

## 3. Visualization

We provide a script to visualize the results of the model also. You can specify the checkpoint path and the data path to visualize the results. The step is quickly similar to evaluation.

```bash
python save.py checkpoint=/home/kin/seflow_best.ckpt dataset_path=/home/kin/data/av2/preprocess_v2/sensor/vis

# The output of above command will be like:
Model: DeFlow, Checkpoint from: /home/kin/model_zoo/v2/seflow_best.ckpt
We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:
python tools/visualization.py --res_name 'seflow_best' --data_dir /home/kin/data/av2/preprocess_v2/sensor/vis
Enjoy! ^v^ ------ 

# Then run the command in the terminal:
python tools/visualization.py --res_name 'seflow_best' --data_dir /home/kin/data/av2/preprocess_v2/sensor/vis
```

https://github.com/user-attachments/assets/f031d1a2-2d2f-4947-a01f-834ed1c146e6


## Cite & Acknowledgements

```
@inproceedings{zhang2024seflow,
  author={Zhang, Qingwen and Yang, Yi and Li, Peizheng and Andersson, Olov and Jensfelt, Patric},
  title={{SeFlow}: A Self-Supervised Scene Flow Method in Autonomous Driving},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  pages={353–369},
  organization={Springer},
  doi={10.1007/978-3-031-73232-4_20},
}
@inproceedings{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Fang, Heng and Geng, Ruoyu and Jensfelt, Patric},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={{DeFlow}: Decoder of Scene Flow Network in Autonomous Driving}, 
  year={2024},
  pages={2105-2111},
  doi={10.1109/ICRA57147.2024.10610278}
}
```

💞 Thanks to RPL member: [Li Ling](https://www.kth.se/profile/liling) helps revise our SeFlow manuscript. Thanks to [Kyle Vedder](https://kylevedder.github.io), who kindly opened his code (ZeroFlow) including pre-trained weights, and discussed their result with us which helped this work a lot. 

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.

❤️: [DeFlow](https://github.com/KTH-RPL/DeFlow), [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval)

