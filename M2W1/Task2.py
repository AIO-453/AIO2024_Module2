import matplotlib.pyplot as plt
import numpy as np


image_path = './M2W1/dog.jpeg'
image = plt.imread(image_path)


def lightness(img):
    # Calculate the max and min values across the RGB channels
    max_rgb = np.max(img, axis=2)
    min_rgb = np.min(img, axis=2)
    # Calculate the lightness
    gray_img = ((max_rgb + min_rgb) / 2)
    return gray_img


def average(img):
    # Calculate the average value across the RGB channels
    avg_rgb = np.mean(img, axis=2)
    return avg_rgb


def luminosity(img):
    # Apply the luminosity formula to convert the image to grayscale
    gray_img = 0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]
    return gray_img


if __name__ == '__main__':
    gray_img_01 = lightness(image)
    gray_img_02 = average(image)
    gray_img_03 = luminosity(image)

    print(gray_img_01[0, 0])
    print(gray_img_02[0, 0])
    print(gray_img_03[0, 0])

    # Hiển thị ảnh trong lưới 2x2
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(gray_img_01, cmap='gray')
    axes[1].imshow(gray_img_02, cmap='gray')
    axes[2].imshow(gray_img_03, cmap='gray')

    plt.axis('off')
    plt.show()
