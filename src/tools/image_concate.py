import argparse
import collections
import cv2

from matplotlib import pyplot as plt

import os 

def concat_image(image_path: str, image_prefix: str, final_results_path: str):
    """
    Concating images back together.
    Images have a certain format x_y. 
    Images with the same x are in the same row. 
    Images should be placed from left to right based on increasing y.
    """
    img_str_list = [i for i in os.listdir(image_path) if (i.find(image_prefix) != -1) and (i.find('.jpg') != -1)]
    
    ## Getting only x_y
    for i in range(len(img_str_list)):
        img_str_list[i] = img_str_list[i].replace((image_prefix + '_'),'')
        img_str_list[i] = img_str_list[i].replace('.jpg','')
    
    image_collections = collections.defaultdict(lambda: [])

    for image_str in img_str_list:
        x, y = image_str.split('_')
        image_collections[x].append(y)
    
    for x in image_collections:
        image_path_list = [os.path.join(image_path,image_prefix + '_' + x + '_' + y + '.jpg') for y in image_collections[x]]
        image_list = [cv2.imread(images) for images in image_path_list]
        

        if int(x) == 0:
            final_img = cv2.hconcat(image_list)
        else:
            subsection_img = cv2.hconcat(image_list)
            final_img = cv2.vconcat([final_img, subsection_img])

    image_name = os.path.join(final_results_path,image_prefix[:-1] + '.jpg')    
    # This is to raise exception that image could not be saved
    if not cv2.imwrite(image_name, final_img):
        raise Exception("Could not write image")
    


def main():
    parser = argparse.ArgumentParser(description='slicing of image')
    parser.add_argument('--image_path', type=str, required=True, help='where the sliced images are located')
    parser.add_argument('--image_prefix', type=str, required=True, nargs='+', help='image prefix = imagelabel_ ie 20210729_132649_')
    parser.add_argument('--final_results_path', type=str, required=True, help='where to store results')

    args=parser.parse_args()
    for image_pre in args.image_prefix:
        concat_image(os.path.join(args.image_path,'exp'), image_pre, args.final_results_path)

    
if __name__ == "__main__":
    main()



