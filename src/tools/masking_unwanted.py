import argparse

import os
import numpy as np
import math
import pandas as pd
import glob

from PIL import Image, ImageOps

def mask_image(image_str, label_str, img_folder_name, lab_folder_name, img_folder_save, lab_folder_save):
    ''' rotates image by 90 degrees-clockwise
    Args:
        image_str (str): string of image name
        label_str (str): string of label name  
        img_folder_name (str): location of original images
        lab_folder_name (str): location of original labels
        image_folder_save (str): location of images to save
        lab_folder_save (str): location of labels to save
    
    Returns:
        (tuple): tuple contraining:
            out: rotated image by 90 degrees
            labels_copy: dataframe for rotated image by 90 degrees
    '''
    #read original image
    image_str_open = img_folder_name + image_str
    img = Image.open(image_str_open)
    image = ImageOps.exif_transpose(img)
    image_np = np.asarray(image)
    
    #read in original label
    labels_str_open = lab_folder_name + label_str
    labels = pd.read_csv(labels_str_open, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    dim_x = image_np.shape[1]
    dim_y = image_np.shape[0]
    
    # Rescale coordinates of labels from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * dim_x
    labels[['y1', 'h']] = labels[['y1', 'h']] * dim_y
    
    # Create masked image to edit on
    masked_image = np.copy(image_np)
    
    # Loop through each row of labels
    for index, row in labels.iterrows():
        # If class of label is masking class
        if row['class'] == 4:
            # Obtain bounding box
            mid_x, mid_y, width_x, width_y = row[1:]
            lower_x = max(0, math.floor( (float(mid_x) - 1/2 * float(width_x))))
            upper_x = min(dim_x, math.floor( (float(mid_x) + 1/2 * float(width_x))))
            lower_y = max(0, math.floor( (float(mid_y) - 1/2 * float(width_y))))
            upper_y = min(dim_y, math.floor( (float(mid_y) + 1/2 * float(width_y))))

            # For each colour channel, find the most common colour value and assign whole bounding box to that colour 
            for i in range(3):
                vals, counts = np.unique(image_np[:,:,i], return_counts=True)
                index = np.argmax(counts)
                masked_image[lower_y:upper_y+1,lower_x:upper_x+1,i] = vals[index]
    
    # Create and save masked image
    out = Image.fromarray(masked_image)
    out_image = img_folder_save + image_str
    out.save(out_image)
    
    # Remove rows with labels 4
    labels_copy = labels[labels["class"] != 4]
    out_label = lab_folder_save + label_str
    labels_copy.to_csv(out_label, header=None, index=None, sep=' ', mode='a')  
    
    return out, labels_copy


img_folder_name = '../data/trial/images/train/'
lab_folder_name = '../data/trial/labels/train/'
img_folder_save = '../data/trial_save/images/train/'
lab_folder_save = '../data/trial_save/labels/train/'
value = 1
img_name = '20210729_131410_0_2.jpg'
lab_name = '20210729_131410_0_2.txt'

