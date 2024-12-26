## :circus_tent: Dataset Preparation

- Download the training Rain13K dataset for constructing the synthetic dataset, run
```
python download_data.py --data train
```

- Download the training/testing SPA dataset for constructing the real-world dataset, refer to the [download link](https://mycuhk-my.sharepoint.com/personal/1155152065_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155152065%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fdataset%2Freal%5Fworld%5Frain%5Fdataset%5FCVPR19&ga=1).


- (Optional) Download the large-scale clean images for the unpaired conditions in diffusion, such as COCO or ImageNet datasets.

- Wrap the paths of the above datasets in ```.flist``` files for unified data loading, run
```
cd Noise-DA
python path_wrapper.py --folder_path 'dataset/derain/train/Rain13K/input' --flist_file './flist_name/derain/rain13k_input.flist'
python path_wrapper.py --folder_path 'dataset/derain/train/Rain13K/target' --flist_file './flist_name/derain/rain13k_gt.flist'
python path_wrapper.py --folder_path 'dataset/derain/train/SPA/input' --flist_file './flist_name/derain/SPA_train.flist' --subfolders
python path_wrapper.py --folder_path 'dataset/derain/test/SPA/input' --flist_file './flist_name/derain/real_test_1000_input.flist'
python path_wrapper.py --folder_path 'dataset/derain/test/SPA/target' --flist_file './flist_name/derain/real_test_1000_gt.flist'
(Optional) python path_wrapper.py --folder_path 'dataset/COCO' --flist_file './flist_name/coco.flist'
```

## :dolphin: Training
- To train the image restoration network with the proposed diffusion loss, customize the training/validation paths (wrapped ```.flist``` files) of datasets in [./configs/options_train.json](./configs/options_train.json) (from 26th to 28th lines and from 39th to 40th lines), and run
```
cd Noise-DA
sh train.sh Deraining/configs/options_train.json
```
- (Optional) To train the image restoration network with the unpaired condition extension, customize the ```"ref_root"``` path of the large-scale clean images (*e.g.*, ```./flist_name/coco.flist```, the 29th line) and set ```"diff_flag"``` to ```2``` (the 6th line) in [./configs/options_train.json](./configs/options_train.json), run
```
sh train.sh Deraining/configs/options_train.json
```

**Note:** The above training scripts use 8 GPUs by default. To use any other number of GPUs, modify the values of ```-gpu``` in [NoiseDA/train.sh](../train.sh).

## :whale: Resume Training from Interruption
- Uncommend the line of ```"resume_state"``` in [./configs/options_train.json](./configs/options_train.json) (17th line) and customize its with the specific resumed path, run
```
sh train.sh Deraining/configs/options_train.json
```

## :framed_picture Evaluation

- Download the pre-trained [models](https://drive.google.com/file/d/1YaEWUTdMIc8S0bn7n-6t6WTLunqf1tG5/view?usp=sharing) and place them in `Noise-DA/checkpoints/` (Ignored if <a href="../README.md## 🏂 Demo & Quick Inference">Demo & Quick Inference</a> has been tried).

- Customize the checkpoint path (14th line) and test dataset path (23rd-24th lines) in [./configs/options_test.json](./configs/options_test.json), run
```
sh test.sh Deraining/configs/options_test.json
```
- To reproduce the quantitative metrics of the SPA dataset, customize the paths of the derained images and ground truth images, run
```
python eva_derain.py
```