# Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration

## Introduction
This is the official implementation for [Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration](https://arxiv.org/abs/2406.18516) (arXiv 2024).

[Kang Liao](https://kangliao929.github.io/), [Zongsheng Yue](https://zsyoaoa.github.io/), [Zhouxia Wang](https://scholar.google.com.hk/citations?user=JWds_bQAAAAJ&hl=zh-CN), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/index.html)

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

Check out more visual results and interactions [here](https://kangliao929.github.io/projects/noise-da/).

## Code
Will be released soon.

## Citation
```bibtex
@article{liao2024denoising,
      title={Denoising as Adaptation: Noise-Space Domain Adaptation for Image Restoration},
      author={Liao, Kang and Yue, Zongsheng and Wang, Zhouxia and Loy, Chen Change},
      journal={arXiv preprint arXiv:2406.18516},
      year={2024}
    }
```

## Contact
For any questions, feel free to email `kang.liao@ntu.edu.sg`.

## License
This project is licensed under [NTU S-Lab License 1.0](LICENSE).
