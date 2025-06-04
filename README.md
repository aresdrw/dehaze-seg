# [Submitted to IEEE TGRS 2025] Dehaze-Seg: Collaboration of Dehazing and Semantic Segmentation: A Multi-task Learning Framework for Hazy Aerial Imagery
[Ziquan Wang]<sup>1</sup>, [Yongsheng Zhang]<sup>1</sup>, [Zimian Wei]<sup>2</sup>, et al. <br />
<sup>1</sup> Information Engineering University  <sup>2</sup> National University of Defense Technology 

Dehaze-seg is a efficient and robust multi-task learning method for dehazing and segmentation task in aerial (UAV) images. We also make two new benchmark with multi-density synthetic fog: HazyUAVid (based on UAVid) and HazyUDD (based on UDD). Our method shows strong generalization on unseen  **HazyDet** and **FoggyDrivingDense**.  
  
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
pip install -r requirements.txt
pip install future tensorboard
```

## Dataset Preparation

```
**HazyUAVid**: Download all image and label packages from [BaiduNetdisk](https://pan.baidu.com/s/1HI17o4Pw4wQ5iybX4Oigdw?pwd=RSDL) and extract them to `datasets/foggy_uavid_for_train`.

**HazyUDD**: Download all image and label packages from [BaiduNetdisk](https://pan.baidu.com/s/1tzA6nB2c5OtpyUvpAfLUHg?pwd=RSDL) and extract them to `datasets/foggy_udd_for_train`.

The final folder structure should look like this:

```
datasets
├── ...
├── foggy_uavid_for_train
│   ├── train
│   ├── val
├── foggy_udd_for_train
│   ├── train
│   ├── val
├── ...
Dehaze_Seg
├── ...
├── configs
├── checkpoints
├── ...
```


## Pretraining Weights
* **Download:** Download HazyUAVid pre-trained weights from [BaiduNetdisk](https://pan.baidu.com/s/1HI17o4Pw4wQ5iybX4Oigdw?pwd=RSDL) and HazyUDD [BaiduNetdisk](https://pan.baidu.com/s/1tzA6nB2c5OtpyUvpAfLUHg?pwd=RSDL) for testing. Unzip them into Dehaze_Seg/work_dirs/ path.

## Dehaze Inference
  Run the evaluation:
  ```
  cd Dehaze_Seg
  python tools/dehaze_inference.py 
  --config ./work_dirs/dehaze-seg_m_udd/alter_freq_udd/dehaze_seg_dehazeformer-M_FreqFusionNeck_mask2former_HazyUDD-alter-512x512.py --checkpoint ./work_dirs/dehaze-seg_m_udd/alter_freq_udd/iter_60000.pth
  --data-dir /path/to/your/data_dir
  --result-dir /path/to/your/result_dir
  ```

You can use the models trained by uavid and udd.

## Seg Inference
  Run the evaluation:
  ```
  cd Dehaze_Seg
  python tools/test.py 
  --config ./work_dirs/dehaze-seg_m_udd/alter_freq_udd/dehaze_seg_dehazeformer-M_FreqFusionNeck_mask2former_HazyUDD-alter-512x512.py --checkpoint ./work_dirs/dehaze-seg_m_udd/alter_freq_udd/iter_60000.pth
  ```

## Acknowledgment
Our implementation is mainly based on following repositories. Thanks for their authors.
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [Rein](https://github.com/w1oves/Rein)
