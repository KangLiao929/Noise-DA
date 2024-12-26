## :circus_tent: Dataset Preparation

- Download the training GoPro dataset for constructing the synthetic dataset, run
```
python download_data.py --data train
```

- Generate image patches from full-resolution training images, run
```
python generate_patches_gopro.py 
```

- Download the training/testing RealBlur_J dataset for constructing the real-world dataset, refer to the [download link](https://cg.postech.ac.kr/research/realblur/).


- (Optional) Download the large-scale clean images for the unpaired conditions in diffusion, such as COCO or ImageNet datasets.

- Wrap the paths of the above datasets in ```.flist``` files for unified data loading, run
```
cd Noise-DA
python path_wrapper.py --folder_path 'dataset/deblur/train/GoPro/input_crops' --flist_file './flist_name/deblur/gopro_train_patch_input.flist'
python path_wrapper.py --folder_path 'dataset/deblur/train/GoPro/target_crops' --flist_file './flist_name/deblur/gopro_train_patch_gt.flist'
(Optional) python path_wrapper.py --folder_path 'dataset/COCO' --flist_file './flist_name/coco.flist'
```

- Split the training and testing sets from the RealBlur_J dataset using the predefined paths in ```./datasets/split_names/```, run
```
cd Deblurring
python data_split.py --source_file_path './datasets/split_names/RealBlur_J_train_list.txt' --output_file_path1 './flist_name/deblur/RealBlur_J_train_gt.flist' --output_file_path2 './flist_name/deblur/RealBlur_J_train_input.flist' --base_path './dataset/deblur/RealBlur_J/'
python data_split.py --source_file_path './datasets/split_names/RealBlur_J_test_list.txt' --output_file_path1 './flist_name/deblur/RealBlur_J_test_gt.flist' --output_file_path2 './flist_name/deblur/RealBlur_J_test_input.flist' --base_path './dataset/deblur/RealBlur_J/'
```

## :dolphin: Training
- To train the image restoration network with the proposed diffusion loss, customize the training/validation paths (wrapped ```.flist``` files) of datasets in [./configs/options_train.json](./configs/options_train.json) (from 26th to 28th lines and from 39th to 40th lines), and run
```
cd Noise-DA
sh train.sh Deblurring/configs/options_train.json
```
- (Optional) To train the image restoration network with the unpaired condition extension, customize the ```"ref_root"``` path of the large-scale clean images (*e.g.*, ```./flist_name/coco.flist```, the 29th line) and set ```"diff_flag"``` to ```2``` (the 6th line) in [./configs/options_train.json](./configs/options_train.json), run
```
sh train.sh Deblurring/configs/options_train.json
```

**Note:** The above training scripts use 8 GPUs by default. To use any other number of GPUs, modify the values of ```-gpu``` in [NoiseDA/train.sh](../train.sh).

## :whale: Resume Training from Interruption
- Uncommend the line of ```"resume_state"``` in [./configs/options_train.json](./configs/options_train.json) (17th line) and customize it with the specific resumed path, run
```
sh train.sh Deblurring/configs/options_train.json
```

## :framed_picture: Evaluation

- Download the pre-trained [models](https://drive.google.com/file/d/1CI1v5M3zLFJ8d_SzgYXlk7Wqa2WMuCF6/view?usp=sharing) and place them in `Noise-DA/checkpoints/` (Ignored if <a href="../README.md## 🏂 Demo & Quick Inference">Demo & Quick Inference</a> has been tried).

- Customize the checkpoint path (14th line) and test dataset path (23rd-24th lines) in [./configs/options_test.json](./configs/options_test.json), run
```
sh test.sh Deblurring/configs/options_test.json
```
- To reproduce the quantitative metrics of the RealBlur_J dataset, customize the paths of the deblurred images and ground truth images, run
```
python eva_deblur.py
```