#!/bin/sh
pasting=1
slicing=1

test_image_src=""
test_image_sliced_raw="./Desktop/DSA4266_Fish_Larva/src/test_images/raw/"
test_image_sliced_predict="./Desktop/DSA4266_Fish_Larva/src/test_images/predict/"
model_weights=""
test_image_num=""

if [ $slicing = 1 ]; then
    python src/tools/tile_yolo.py \
    -source $test_image_src
    -falsefolder $test_image_sliced_raw 
    -ext .jpg -size 640
fi

python src/yolov3-master/detect.py \
--weights $model_weights
--source $test_image_sliced_raw
--project $test_image_sliced_predict
--save-txt

if [ $pasting = 1 ]; then
    python src/tools/image_concate.py \
    --image_path $test_image_sliced_predict \
    --image_prefix $test_image_num
fi