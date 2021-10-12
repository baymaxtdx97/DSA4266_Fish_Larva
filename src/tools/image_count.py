import argparse
import os
import collections

def image_class_counter(image_labels_path: str , image_prefix: str, final_results_path: str):
    """
    Taking in all txt files of an imaged and compiling them together
    :param image_labels_path: directory to find the image txt files
    :param image_prefix: image label 20210729_132649
    :final_results_path: where to save the final compiled txt
    """
    doc_list = os.listdir(image_labels_path)
    image_list = [image_txt for image_txt in doc_list if image_prefix in image_txt]

    final_counts = collections.Counter()

    for image_file in image_list:
        with open(os.path.join(image_labels_path,image_file),"r") as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            class_list = collections.Counter([int(i.split(' ')[0]) for i in lines])
            final_counts.update(class_list)
    
    sorted_list = sorted(final_counts.items(),key = lambda i: i[0])

    image_proper_labels = {0: 'Fertilised Eggs', 1:'Unfertilised Eggs', 2: 'Fish Larvae', 3: 'Unidentifiable'}

    with open(os.path.join(final_results_path, image_prefix +'.txt'),'w+') as f:
        for k,v in sorted_list:
            f.write("{} {}\n".format(image_proper_labels[k],v))



def main():
    parser = argparse.ArgumentParser(description='slicing of image')
    parser.add_argument('--image_labels_path', type=str, required=True, help='where the prediction txt files are located')
    parser.add_argument('--image_label_prefix', type=str, required=True, nargs='+', help='image prefix = imagelabel ie 20210729_132649')
    parser.add_argument('--final_results_path', type=str, required=True, help='where to store results')

    args=parser.parse_args()

    for image_labels in args.image_label_prefix:
        image_class_counter(args.image_labels_path, image_labels, args.final_results_path)


if __name__ == "__main__":
    main()

