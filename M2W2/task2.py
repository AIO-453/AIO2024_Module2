import numpy as np
import cv2

bg1_image = cv2.imread('GreenBackground.png', 1)
bg1_image = cv2.resize(bg1_image, (678, 381))

ob_image = cv2.imread('Object.png', 1)
ob_image = cv2.resize(ob_image, (678, 381))

bg2_image = cv2.imread('NewBackground.jpg', 1)
bg2_image = cv2.resize(bg2_image, (678, 381))


def compute_difference(bg_img, input_img):
    # bg_img = bg_img.astype(np.float64)
    # input_img = input_img.astype(np.float64)
    difference_single_channel = input_img-bg_img
    difference_single_channel = np.clip(difference_single_channel, 0, 255)
    difference_single_channel = difference_single_channel.astype(np.uint8)
    return difference_single_channel


def compute_binary_mask(image, threshold=5):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    _, binary_mask = cv2.threshold(
        gray_image, threshold, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    return binary_mask


def replace_background(bg1_image, bg2_image, ob_image):
    difference_singgle_channel = compute_difference(bg1_image, ob_image)
    binary_mask = compute_binary_mask(difference_singgle_channel)
    output = np.where(binary_mask == 255, ob_image, bg2_image)
    return output


difference_single_channel = compute_difference(bg1_image, ob_image)
binary_mask = compute_binary_mask(difference_single_channel)
output = replace_background(bg1_image, bg2_image, ob_image)


if __name__ == '__main__':
    cv2.imshow('binary_mask', binary_mask)
    cv2.imshow('difference_single_channel', difference_single_channel)
    cv2.imshow('output', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
