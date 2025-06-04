# [Submitted to IEEE TGRS 2024] Dehaze-Seg: Collaboration of Dehazing and Semantic Segmentation: A Multi-task Learning Framework for Hazy Aerial Imagery
[Ziquan Wang]<sup>1</sup>, [Yongsheng Zhang]<sup>1</sup>, [Zimian Wei]<sup>2</sup>, et al. <br />
<sup>1</sup> Information Engineering University  <sup>2</sup> National University of Defense Technology 

Dehaze-seg is a efficient and robust multi-task learning method for dehazing and segmentation task in aerial (UAV) images. We also make two new benchmark with multi-density synthetic fog: HazyUAVid (based on UAVid) and HazyUDD (based on UDD). Our method shows strong generalization on unseen  **HazyDet** and **FoggyDrivingDense**.  
![Dehaze-seg Framework](docs/framework.png)
![HazyUAVid and HazyUDD synthetic foggy benchmarks](docs/hazyuavid_hazyudd.png)

## Powerful Dehaze Visualization
Trained on HazyUAVid dataset, we can get the dehazing .

![On the unseen aerial HazyDet](docs/dehaze_hazydet.png)
![On the HazyUAVid and HazyUDD benchmarks](docs/dehaze_hazyuavid_hazyudd.jpg)
![On the unseen unseen car-view Foggy Driving Dense](docs/dehaze_car.jpg)

## Segmentation Visualization
Trained on HazyUAVid (8 classes) and HazyUDD (6 classes).

![HazyUAVid](docs/hazyuavid.jpg)
  
## Environment Setup
To set up your environment, execute the following commands:
```bash
conda create -n rein -y
conda activate dehaze-seg
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20' # optional for DINOv2
pip install -r requirements.txt
pip install future tensorboard
```

## Dataset Preparation

```
**HazyUAVid**: Download all image and label packages from [BaiduNetdisk](https://acdc.vision.ee.ethz.ch/)Extraction code and extract them to `datasets/foggy_uavid_for_train`.

**HazyUDD**: Download all image and label packages from [BaiduNetdisk](http://www.urbansyn.org/#loaded)Extraction code and extract them to `datasets/foggy_udd_for_train`.

The final folder structure should look like this:

```
datasets
|--foggy_uavid_for_train
    |--train
    |--val
|--foggy_udd_for_train
    |--train
    |--val
|--...
Dehaze_Seg
├--configs
|--checkpoints
   |--dehazeformer-s.pth
   |--dehazeformer-m.pth
|--work_dirs
...
```
## Pretraining Weights
* **Download:** Download pre-trained weights from [facebookresearch](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) for testing. Place them in the project directory without changing the file name.
## Evaluation
  Run the evaluation:
  ```
  python tools/test.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py checkpoints/dinov2_rein_and_head.pth --backbone dinov2_converted.pth
  ```
  For most of provided release checkpoints, you can run this command to evluate
  ```
  python tools/test.py /path/to/cfg /path/to/checkpoint --backbone /path/to/dinov2_converted.pth #(or dinov2_converted_1024x1024.pth)
  ```

## Training
Start training in single GPU:
```
python tools/train.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py --work-dir /path/to/your/work_dir
```
Start training in multiple GPU:
```
PORT=12345 CUDA_VISIBLE_DEVICES=1,2,3,4 bash tools/dist_train.sh configs/dinov2/rein_dinov2_mask2former_1024x1024_bs4x2.py NUM_GPUS
```
* Q: How to Visualize?
  * A: Use `tools/visualize.py`, such as :
  ``` bash
  python tools/visualize.py /path/to/cfg /path/to/checkpoint /path/to/images --backbone /path/to/converted_backbone
  ```
  here `/path/to/images` can be a filename or image folder.


## Acknowledgment
Our implementation is mainly based on following repositories. Thanks for their authors.
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [Rein](https://github.com/w1oves/Rein)