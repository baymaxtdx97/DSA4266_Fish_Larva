import argparse

import os
import numpy as np
import math
import pandas as pd
import glob

from PIL import Image, ImageOps

def rotate_image_ninety(image_str, label_str, img_folder_name, lab_folder_name, image_folder_save, lab_folder_save):
    ''' rotates image by 90 degrees-clockwise
    Args:
        image (str): string of image name
        label (str): string of label name  
        img_folder_name (str): location of original images
        lab_folder_name (str): location of original labels
        image_folder_save (str): location of images to save
        lab_folder_save (str): location of labels to save
    
    Returns:
        (tuple): tuple contraining:
            out_image: rotated image by 90 degrees
            labels_copy: dataframe for rotated image by 90 degrees
    '''
    #read original image
    image_str_open = img_folder_name + image_str
    img = Image.open(image_str_open)
    image = ImageOps.exif_transpose(img)
    image_np = np.asarray(image)
    
    #rotate image
    out = img.rotate(90, expand=True)
    out_image = img_folder_save + image_str 
    out.save(out_image)
    
    #read in original label
    labels_str_open = lab_folder_name + label_str
    labels = pd.read_csv(labels_str_open, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    dim_x = image_np.shape[1]
    dim_y = image_np.shape[0]
    # Rescale coordinates of labels from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * dim_x
    labels[['y1', 'h']] = labels[['y1', 'h']] * dim_y
    
    #change labels 
    labels_copy = labels.copy()
    for i in range(0, len(labels)):
        row = labels.iloc[i]
        ## original labels
        mid_x, mid_y, width_x, width_y = row[1:]
        new_mid_x = mid_y
        new_mid_y = dim_x - mid_x
        new_width_x = width_y
        new_width_y = width_x
        new_row = [int(row[0]), new_mid_x, new_mid_y, new_width_x, new_width_y]
        labels_copy.iloc[i]=new_row
    
    # Rescale coordinates of labels_copy back to original
    labels_copy[['x1', 'w']] = labels_copy[['x1', 'w']] / dim_y
    labels_copy[['y1', 'h']] = labels_copy[['y1', 'h']] / dim_x
    out_label = lab_folder_save+ label_str
    labels_copy.to_csv(out_label, header=None, index=None, sep=' ', mode='a')    
    return out_image, labels_copy

def rotate_image_180(image_str, label_str, img_folder_name, lab_folder_name, image_folder_save, lab_folder_save):
    ''' rotates image by 180 degrees-clockwise
    Args:
        image (str): string of image name
        label (str): string of label name  
        img_folder_name (str): location of original images
        lab_folder_name (str): location of original labels
        image_folder_save (str): location of images to save
        lab_folder_save (str): location of labels to save 
    
    Returns:
        (tuple): tuple contraining:
            out_image: rotated image by 180 degrees
            labels_copy: dataframe for rotated image by 180 degrees
    '''
    #read original image
    image_str_open = img_folder_name + image_str
    img = Image.open(image_str_open)
    image = ImageOps.exif_transpose(img)
    image_np = np.asarray(image)
    
    #rotate image
    out = img.rotate(180, expand=True)
    out_image = img_folder_save+ image_str
    out.save(out_image)
    
    #read in original label
    labels_str_open = lab_folder_name + label_str
    labels = pd.read_csv(labels_str_open, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    dim_x = image_np.shape[1]
    dim_y = image_np.shape[0]
    # Rescale coordinates of labels from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * dim_x
    labels[['y1', 'h']] = labels[['y1', 'h']] * dim_y
    
    #change labels 
    labels_copy = labels.copy()
    for i in range(0, len(labels)):
        row = labels.iloc[i]
        ## original labels
        mid_x, mid_y, width_x, width_y = row[1:]
        new_mid_x = dim_x - mid_x
        new_mid_y = dim_y - mid_y
        new_width_x = width_x
        new_width_y = width_y
        new_row = [int(row[0]), new_mid_x, new_mid_y, new_width_x, new_width_y]
        labels_copy.iloc[i]=new_row
    
    # Rescale coordinates of labels_copy back to original
    labels_copy[['x1', 'w']] = labels_copy[['x1', 'w']] / dim_x
    labels_copy[['y1', 'h']] = labels_copy[['y1', 'h']] / dim_y
    out_label = lab_folder_save + label_str
    labels_copy.to_csv(out_label, header=None, index=None, sep=' ', mode='a')    
    return out_image, labels_copy

def flip_image(image_str, label_str, img_folder_name, lab_folder_name, image_folder_save, lab_folder_save):
    ''' flips image to create mirror image
    Args:
        image (str): string of image name
        label (str): string of label name  
        img_folder_name (str): location of original images
        lab_folder_name (str): location of original labels
        image_folder_save (str): location of images to save
        lab_folder_save (str): location of labels to save
    
    Returns:
        (tuple): tuple contraining:
            out_image: flipped image along the vertical axis
            labels_copy: dataframe for flipped image along the vertical axis
    '''
    #read original image
    image_str_open = img_folder_name + image_str
    img = Image.open(image_str_open)
    image = ImageOps.exif_transpose(img)
    image_np = np.asarray(image)
    
    #rotate image
    out = ImageOps.mirror(img)
    out_image = img_folder_save + image_str
    out.save(out_image)
    
    #read in original label
    labels_str_open = lab_folder_name + label_str
    labels = pd.read_csv(labels_str_open, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    dim_x = image_np.shape[1]
    dim_y = image_np.shape[0]
    # Rescale coordinates of labels from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * dim_x
    labels[['y1', 'h']] = labels[['y1', 'h']] * dim_y
    
    #change labels 
    labels_copy = labels.copy()
    for i in range(0, len(labels)):
        row = labels.iloc[i]
        ## original labels
        mid_x, mid_y, width_x, width_y = row[1:]
        new_mid_x = dim_x - mid_x
        new_mid_y = mid_y
        new_width_x = width_x
        new_width_y = width_y
        new_row = [int(row[0]), new_mid_x, new_mid_y, new_width_x, new_width_y]
        labels_copy.iloc[i]=new_row
    
    # Rescale coordinates of labels_copy back to original
    labels_copy[['x1', 'w']] = labels_copy[['x1', 'w']] / dim_x
    labels_copy[['y1', 'h']] = labels_copy[['y1', 'h']] / dim_y
    out_label = lab_folder_save + label_str
    labels_copy.to_csv(out_label, header=None, index=None, sep=' ', mode='a')  
    return out_image, labels_copy

def overall_augmentation(random_number, image_str, label_str, \
                         img_folder_name, lab_folder_name, image_folder_save, lab_folder_save):
    ''' flips image to create mirror image
    Args:
        random_number (int): int either 1, 2 or 3 to decide which augmentation
            technique to use
        image (str): string of image name
        label (str): string of label name  
        img_folder_name (str): location of original images
        lab_folder_name (str): location of original labels
        image_folder_save (str): location of images to save
        lab_folder_save (str): location of labels to save
    
    Returns:
        (tuple): tuple contraining:
            out_image: flipped image along the vertical axis
            labels_copy: dataframe for flipped image along the vertical axis
    '''
    if random_number == 1:
        rotate_image_ninety(image_str, label_str, img_folder_name, lab_folder_name, image_folder_save, lab_folder_save)
    elif random_number == 2:
        rotate_image_180(image_str, label_str, img_folder_name, lab_folder_name, image_folder_save, lab_folder_save) 
    else:
        flip_image(image_str, label_str, img_folder_name, lab_folder_name, image_folder_save, lab_folder_save)


                                  
