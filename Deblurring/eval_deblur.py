import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import cv2
import argparse

def calculate_psnr_ssim(args):
    psnr_values = []
    ssim_values = []

    images1 = sorted(os.listdir(args.input_path))
    images2 = sorted(os.listdir(args.gt_path))

    assert len(images1) == len(images2), "The number of images in each folder must be the same."

    for img1, img2 in zip(images1, images2):
        image1 = cv2.imread(os.path.join(args.input_path, img1))
        image2 = cv2.imread(os.path.join(args.gt_path, img2))

        psnr = peak_signal_noise_ratio(image1, image2)
        ssim = structural_similarity(image1, image2, channel_axis=2)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input/')
    parser.add_argument('--gt_path', type=str, default='./gt/')

    print('<==================== Evaluating ===================>\n')

    args = parser.parse_args()
    avg_psnr, avg_ssim = calculate_psnr_ssim(args)
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")