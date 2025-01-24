# Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration (ICLR 2025)

## Introduction
This is the official implementation for [Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration](https://arxiv.org/abs/2406.18516) (arXiv version).

[Kang Liao](https://kangliao929.github.io/), [Zongsheng Yue](https://zsyoaoa.github.io/), [Zhouxia Wang](https://wzhouxiff.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/index.html)

S-Lab, Nanyang Technological University


<div align="center">
  <img src="https://github.com/KangLiao929/Noise-DA/blob/main/assets/new-tesear.png" height="340">
</div>

> ### Why Noise-Space Domain Adaptation?
> Existing domain adaptation approaches are mainly developed for high-level vision tasks. However, aligning high-level deep representations in *feature space* may overlook low-level variations essential for image restoration, while *pixel-space* approaches often involve computationally intensive adversarial paradigms that can lead to instability during training. In this work, we propose a new *noise-space* solution that preserves low-level appearance across different domains within a compact and stable framework.
>  ### Features
>  * Our work represents the first attempt at addressing domain adaptation in the noise space for image restoration. We show the unique benefits from *diffusion loss* in eliminating the gap between the synthetic and real-world data, which cannot be achieved using existing losses.
>  * To eliminate the shortcut learning in joint training, we design strategies to fool the diffusion model, making it difficult to distinguish between synthetic and real conditions, thereby encouraging both to align consistently with the target clean distribution.
>  * Our method offers a general and flexible adaptation strategy applicable beyond specific restoration tasks. It requires no prior knowledge of noise distribution or degradation models and is compatible with various restoration networks. The diffusion model is discarded after training, incurring no extra computational cost during restoration inference.

## 📝 Changelog & News

- [x] 2024.10.08: The project page of Noise-DA is online.
- [x] 2024.12.26: Release the code (both training and inference) and pre-trained models.
- [x] 2025.01.23: This paper has been accepted to ICLR 2025.
- [ ] Release more pre-trained restoration models of our extended experiments, such as DnCNN, Uformer, SwinIR, Restormer, etc.
- [ ] Release Gradio Demo.

## :desktop_computer: Requirements and Installation
The code has been implemented with PyTorch 2.1.2 and CUDA 12.1.

An example of installation commands is provided as follows:

```
# git clone this repository
git clone https://github.com/KangLiao929/Noise-DA
cd Noise-DA

# create new anaconda env
conda create -n Noise-DA python=3.9
conda activate Noise-DA

# install python dependencies
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 🏂 Demo & Quick Inference
- Download the [pretrained models](https://drive.google.com/drive/folders/1H-cdUzW7nkw3MBNi9iliXKjwj_xjpjcZ?usp=sharing) of each image restoration task (*e.g.*, denoising, deraining, and deblurring) to the ```checkpoints``` folder.

- Customize the paths of the pretrained models ```"checkpoint"``` and degraded images ```"data_root"``` in ```.json``` of the [`configs_demo`](./configs_demo) folder. We also provide some examples of degraded images in the [`inputs`](./inputs) folder. Run the following scripts for different restoration tasks.

```
# test the image denoising model
sh test.sh ./configs_demo/denoising.json

# test the image deraining model
sh test.sh ./configs_demo/deraining.json

# test the image deblurring model
sh test.sh ./configs_demo/deblurring.json
```
The restored results can be found in the [`results`](./results) folder. Note that the above restoration networks are built based on the classical and handy U-Net architecture. Better restoration performance can be achieved using more powerful archiectures such as SwinIR and Restormer, and we will release their pretrained models soon.

🌈 Check out more visual results and restoration interactions [here](https://kangliao929.github.io/projects/noise-da/).

## :airplane: Training and Evaluation
The instructions of the dataset preparation, training, and evaluation (reproduce our quantitative metrics) for each image restoration task, are provided in their respective directories. Here is a summary table containing hyperlinks for easy navigation:

<table>
  <tr>
    <th align="left">Task</th>
    <th align="center">Dataset Instructions</th>
    <th align="center">Training Instructions</th>
    <th align="center">Evaluation Instructions</th>
  </tr>
  <tr>
    <td align="left">Image Denoising</td>
    <td align="center"><a href="Denoising/README.md## :circus_tent: Dataset Preparation">Link</a></td>
    <td align="center"><a href="Denoising/README.md## :dolphin: Training">Link</a></td>
    <td align="center"><a href="Denoising/README.md## :framed_picture Evaluation">Link</a></td>
  </tr>
  <tr>
    <td>Image Deraining</td>
    <td align="center"><a href="Deraining/README.md## :circus_tent: Dataset Preparation">Link</a></td>
    <td align="center"><a href="Deraining/README.md## :dolphin: Training">Link</a></td>
    <td align="center"><a href="Deraining/README.md## :framed_picture Evaluation">Link</a></td>
  </tr>
  <tr>
    <td>Image Deblurring</td>
    <td align="center"><a href="Deblurring/README.md## :circus_tent: Dataset Preparation">Link</a></td>
    <td align="center"><a href="Deblurring/README.md## :dolphin: Training">Link</a></td>
    <td align="center"><a href="Deblurring/README.md## :framed_picture Evaluation">Link</a></td>
  </tr>
</table>

## :newspaper_roll: License
This project is licensed under [NTU S-Lab License 1.0](LICENSE). Redistribution and use should follow this license.

## :clap: Acknowledgement
This project is based on [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models), [openai/guided-diffusion](https://github.com/openai/guided-diffusion), and [Restormer](https://github.com/swz30/Restormer). Thanks for their awesome works.

## :thumbsup: Citation
If you find our work useful for your research, please consider citing the paper:
```bibtex
@article{liao2024denoising,
      title={Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration},
      author={Liao, Kang and Yue, Zongsheng and Wang, Zhouxia and Loy, Chen Change},
      journal={arXiv preprint arXiv:2406.18516},
      year={2024}
    }
```

## :phone: Contact
For any questions, feel free to email `kang.liao@ntu.edu.sg`.