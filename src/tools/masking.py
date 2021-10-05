from PIL import Image, ImageOps
import numpy as np
import math
import pandas as pd
import glob
import os

# Specify folder names for image and labels
img_folder_name = 'Fish_Larva_Count_split/images/val'
lab_folder_name = 'Fish_Larva_Count_split/labels/val'
ext = '.jpg'

# Read in all images within folder and split to remove path
imnames = glob.glob(f'{img_folder_name}/*{ext}')
img_names = [name.replace('\\', '/').split('/')[-1] for name in imnames]
os.makedirs(img_folder_name + "/masked") # Make directory for saving masked images

# Loop through each image
for img_name in img_names:
    # Label name
    labname = img_name.replace(ext, '.txt')

    # Read in image, make sure orientation is correct
    image = Image.open(img_folder_name + "/" + img_name)
    image = ImageOps.exif_transpose(image)
    image_np = np.asarray(image)

    # Read in labels as a dataframe
    with open(lab_folder_name + "/" + labname) as f:
        labels = pd.read_csv(f, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

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
    image2 = Image.fromarray(masked_image)
    image2.save(img_folder_name + '/masked/' + img_name)

        
