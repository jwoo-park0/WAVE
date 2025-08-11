<p align="center">
  <h1 align="center">🌊WAVE: Warp-Based View Guidance for Consistent Novel View Synthesis Using a Single Image</h1>
  <h3 align="center"><b>ICCV 2025</b></h3>
  <p align="center">
    <h3 align="center">
      <a href="mailto:wldn1677@yonsei.ac.kr"><strong>Jiwoo Park</strong></a> · 
      <a href="mailto:teunchoi@yonsei.ac.kr"><strong>Tae Eun Choi</strong></a> · 
      <a href="mailto:youngjun@yonsei.ac.kr"><strong>Youngjun jun</strong></a> ·  
      <a href="mailto:seongjae@yonsei.ac.kr"><strong>Seong Jae Hwang</strong></a>
    </h3>
    <h3 align="center">
      Yonsei University
    </h3>
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2506.23518"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2411.17150-b31b1b.svg"></a>
    <a href="https://jwoo-park0.github.io/wave.github.io/"><img alt='Project Page' src="https://img.shields.io/badge/Project-Website-orange"></a>
  </p>
  <br>
</p>
 
<be>
<img width="1775" alt="Image" src="https://jwoo-park0.github.io/wave.github.io/static/images/main.png" />

> #### **WAVE: Warp-Based View Guidance for Consistent Novel View Synthesis Using a Single Image**<be>  
>IEEE/CVF International Conference on Computer Vision (ICCV) 2025  
>Jiwoo Park, Tae Eun Choi, Youngjun Jun, Seong Jae Hwang  
>Yonsei University
### Abstract
Generating high-quality novel views of a scene from
a single image requires maintaining structural coherence
across different views, referred to as view consistency. While
diffusion models have driven advancements in novel view
synthesis, they still struggle to preserve spatial continuity
across views. Diffusion models have been combined with 3D
models to address the issue, but such approaches lack efficiency due to their complex multi-step pipelines. This paper
proposes a novel view-consistent image generation method
which utilizes diffusion models without additional modules.
Our key idea is to enhance diffusion models with a trainingfree method that enables adaptive attention manipulation
and noise reinitialization by leveraging view-guided warping to ensure view consistency. Through our comprehensive
metric framework suitable for novel-view datasets, we show
that our method improves view consistency across various
diffusion models, demonstrating its broader applicability.

## :book: Contents
<!--ts-->
   * [Installation](#installation)
   * [Usage](#usage)
   * [WAVE](#wave)
      * [Overall Pipeline of WAVE](#overall-pipeline-of-wave)
      * [Qualitative Results](#qualitative-results)
      * [Quantitative  Results](#quantitative-results)
   * [Evaluation](#evaluation)
   * [Acknowledgements](#acknowledgements)
      <!-- * [Citation](#citation) -->

<!--te-->


## Installation


### Requirements
Our code builds on the requirement of the official [MegaScenes](https://github.com/MegaScenes/nvs). To set up their environment, please run:
```shell script
conda env create -n wave python=3.8
conda activate wave 
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit --yes
pip install -r requirements.txt
```

### Download Checkpoints and Datasets
To get started, download the [MegaScenes checkpoint](https://github.com/MegaScenes/nvs?tab=readme-ov-file) (with warps and poses) and [MegaScenes dataset](https://github.com/MegaScenes/dataset) using the command below:
```shell script
## model checkpoints
s5cmd --no-sign-request cp s3://megascenes/nvs_checkpoints/warp_plus_pose/iter_112000/* ./configs/warp_plus_pose/iter_112000/
## MegaScenes datasets
s5cmd --no-sign-request cp s3://megascenes/images/* ./MegaScenes/images/
```
For the RE10K dataset, please follow the instructions provided
[Re10K](https://google.github.io/realestate10k/) and 
[Re10K downloader](https://github.com/cashiwamochi/RealEstate10K_Downloader?tab=readme-ov-file)

## Usage
To generate novel views images, you can simply run the `inference.py` script. For example, 
```shell script
python inference.py -e configs/ -r 112000 -config_name config
                    \ -i data/examples/caernarfon_castle.jpg
                    \ -s save_dir SAVE_DIR
``` 
Notes:
- To run WAVE, you need to specify the location of the config file like `-e configs/` and model's training iteration `-r 112000`.
- You must set the name of configuration you want to use: `-config_name config`.
- When providing an input image, use the following format: `-i data/examples/caernarfon_castle.jpg`.
- To specify the directory where the output files would be saved, use: `-s [SAVE_DIR]`.

All images will be saved to the path `save_dir`. Warped images will be stored in the `warped` subfolder. Each view’s images will be stored in the `per_view_generation` subfolder. Continuous videos generated from each view will be stored in the `videos` subfolder.


## WAVE 

### Overall Pipeline of WAVE 
Given an input view, depth map and continuous camera poses, our method generates scene images with
smooth view transitions through three distinct processes: (a) adaptive warp-range selection utilizes warped region masks, from which
the relevance between viewpoints is determined to compute the reference range for attention. (b) pose-aware noise initialization (PANI)
re-initializes the diffusion initial noise by leveraging warped images and initial noise, incorporating frequency domain information. (c)
warp-guided adaptive attention (WGAA) utilizes the warped region masks and reference range obtained from the adaptive warp-range
selection, performing masked batch self-attention.
<div align="center">
  <img width="800" alt="spectral" src="https://jwoo-park0.github.io/wave.github.io/static/images/pipeline.png">
</div>




### Qualitative Results
Comparison of qualitative results on MegaScenes, Re10K, and DTU using our method and the baselines: MegaScenes and VistaDream.
<div align="center">
  <img width="900" alt="main_results" src="https://jwoo-park0.github.io/wave.github.io/static/images/qualitative.png">
</div>

### Quantitative Results
- Quantitative comparison of our method WAVE with diffusion-based novel view synthesis models on three datasets.
<div align="center">
  <img width="900" alt="quan_results" src="https://jwoo-park0.github.io/wave.github.io/static/images/table.png">
</div>


- Quantitative comparison of our method WAVE with other novel view synthesis models, considering both inference time and consistency metrics.
<div align="center">
  <img width="450" alt="quan_results" src="https://jwoo-park0.github.io/wave.github.io/static/images/inference_time.png">
</div>

## Evaluation
To be released soon.


## Acknowledgements
This repository has been developed based on the [MegaScenes](https://github.com/MegaScenes/nvs) repository. Thanks for the great work!