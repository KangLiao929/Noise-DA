## :circus_tent: Dataset Preparation

- Download training (DIV2K, Flickr2K, WED, BSD) datasets for constructing the synthetic dataset, run
```
python download_data.py --data train --noise gaussian
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_dfwb.py 
```

- Download the SIDD training/validation data for constructing the real-world dataset, run
```
python download_data.py --data train --noise real
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_sidd.py 
```

- Download the SIDD test data, run 
```
python download_data.py --noise real --data test --dataset SIDD
```

- (Optional) Download the large-scale clean images for the unpaired conditions in diffusion, such as COCO or ImageNet datasets.

- Wrap the paths of the above datasets in ```.flist``` files for unified data loading, run
```
cd Noise-DA
python path_wrapper.py --folder_path 'dataset/denoise/train/DFWB/input_crops' --flist_file './flist_name/denoise/DFWB.flist'
python path_wrapper.py --folder_path 'dataset/denoise/train/SIDD/input_crops' --flist_file './flist_name/denoise/SIDD_train_input.flist'
python path_wrapper.py --folder_path 'dataset/denoise/val/SIDD/input_crops' --flist_file './flist_name/denoise/SIDD_val_input.flist'
python path_wrapper.py --folder_path 'dataset/denoise/val/SIDD/target_crops' --flist_file './flist_name/denoise/SIDD_val_gt.flist'
(Optional) python path_wrapper.py --folder_path 'dataset/COCO' --flist_file './flist_name/coco.flist'
```

## :dolphin: Training
- To train the image restoration network with the proposed diffusion loss, customize the training/validation paths (wrapped ```.flist``` files) of datasets in [./configs/options_train.json](./configs/options_train.json) (from 26th to 28th lines and from 39th to 40th lines), and run
```
cd Noise-DA
sh train.sh Denoising/configs/options_train.json
```
- (Optional) To train the image restoration network with the unpaired condition extension, customize the ```"ref_root"``` path of the large-scale clean images (*e.g.*, ```./flist_name/coco.flist```, the 29th line) and set ```"diff_flag"``` to ```2``` (the 6th line) in [./configs/options_train.json](./configs/options_train.json), run
```
sh train.sh Denoising/configs/options_train.json
```

**Note:** The above training scripts use 8 GPUs by default. To use any other number of GPUs, modify the values of ```-gpu``` in [NoiseDA/train.sh](../train.sh).

## :whale: Resume Training from Interruption
- Uncommend the line of ```"resume_state"``` in [./configs/options_train.json](./configs/options_train.json) (17th line) and customize its with the specific resumed path, run
```
sh train.sh Denoising/configs/options_train.json
```

## :framed_picture Evaluation

- Download the pre-trained [models](https://drive.google.com/file/d/1ZJ7LXCfQptjqCn5PYWzZTsRZJW501CjC/view?usp=sharing) and place them in `Noise-DA/checkpoints/` (Ignored if <a href="../README.md## 🏂 Demo & Quick Inference">Demo & Quick Inference</a> has been tried).

- Customize the checkpoint path (14th line) and test dataset path (23rd-24th lines) in [./configs/options_test.json](./configs/options_test.json), run
```
sh test.sh Denoising/configs/options_test.json
```
- To reproduce the quantitative metrics of the SIDD dataset, customize the paths of the denoised images (*i.e.*, ```Idenoised.mat```) and ground truth images (*i.e.*, ```ValidationGtBlocksSrgb.mat```) and run the following script in MATLAB.
```
eva_denoise.m
```