import os
import glob

# Specify folder names for labels
lab_folder_name = 'Fish_Larva_Count_split/labels/val'
ext = '.txt'

# Read in all labels within folder and split to remove path
labnames = glob.glob(f'{lab_folder_name}/*{ext}')
lab_names = [name.replace('\\', '/').split('/')[-1] for name in labnames]
os.makedirs(lab_folder_name + "/removed_mask") # Make directory for saving new labels

# For each image label
for lab_name in lab_names:

    # Read in labels and remove masking bounding boxes
    new = []
    with open(lab_folder_name + '/' + lab_name) as f:
        lines = f.read().splitlines()
        for line in lines:
            if line[0] != '4':
                new.append(line)            
    
    # Save new labels
    with open(lab_folder_name + "/removed_mask/" + lab_name, 'w') as f:
        f.write('\n'.join(new))
