import os
import pandas as pd

image_names = []
all_text_files = os.listdir("labels")

for label_file in all_text_files:
    label_file_name = label_file.split("_")[2] + "_" + label_file.split("_")[3][:-4]
    if label_file_name not in image_names:
        image_names.append(label_file_name)

print(image_names)
unfertilized_egg_counts_lst = []
fertilized_egg_counts_lst = []
fish_larvae_counts_lst = []

for image_name in image_names:
    #print(image_name)
    image_txt_files = [text_file for text_file in all_text_files if text_file[4:-4] == image_name]
    #print(image_txt_files)
    unfertilized_egg_counts = 0
    fertilized_egg_counts = 0
    fish_larvae_counts = 0
    for txt_file in image_txt_files:
        with open('labels/'+txt_file) as f:
            labels = f.read().splitlines()
        for label in labels:
            if label[0] == '0':
                unfertilized_egg_counts += 1
            elif label[0] == '1':
                fertilized_egg_counts += 1
            elif label[0] == '2':
                fish_larvae_counts += 1
                
    unfertilized_egg_counts_lst.append(unfertilized_egg_counts)
    fertilized_egg_counts_lst.append(fertilized_egg_counts)
    fish_larvae_counts_lst.append(fish_larvae_counts)



results = pd.DataFrame({'image_name' : image_names,
                       'unfertilized_eggs' : unfertilized_egg_counts_lst,
                       'fertilized_eggs' : fertilized_egg_counts_lst,
                       'fish_larvae' : fish_larvae_counts_lst})
print(results)

print("Total number of unfertilized eggs", sum(unfertilized_egg_counts_lst))
print("Total number of fertilized eggs", sum(fertilized_egg_counts_lst))
print("Total number of fish larvae", sum(fish_larvae_counts_lst))

print("2/10th of total unfertilized egg = ", sum(unfertilized_egg_counts_lst)/5)
print("2/10th of total fertilized egg = ", sum(fertilized_egg_counts_lst)/5)
print("2/10th of total fish larvae = ", sum(fish_larvae_counts_lst)/5)
    

