import os
import cv2
import numpy as np

def average_images(directory):
    # Get a list of all .png image files in the directory
    image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png')]

    # Load first image to create an accumulator
    image_accumulator = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE).astype(np.float64)

    # Iterate over remaining images, adding each one to the accumulator
    for file in image_files[1:]:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        image_accumulator += image

    # Divide by the number of images to get the average
    average_image = image_accumulator / len(image_files)

    return average_image.astype(np.uint8) # Converting back to 8-bit format

average_img = average_images('./results2/outputs2')

# To save the averaged image
cv2.imwrite('./results2/test_enaverage_image.png', average_img)