#!/bin/sh
pasting=1
slicing=1
predicting=1
image_count=1

test_image_src="/c/Users/Dell/Desktop/DSA4266/test_images/20210903_095054.jpg"
test_image_sliced_raw="/c/Users/Dell/Desktop/DSA4266/test_images_sliced_raw"
test_image_sliced_predict="/c/Users/Dell/Desktop/DSA4266/test_images_sliced_predict"
test_image_sliced_predict_labels="/c/Users/Dell/Desktop/DSA4266/test_images_sliced_predict/exp/labels"
model_weights="/c/Users/Dell/Desktop/DSA4266/best.pt"
final_results_stored="/c/Users/Dell/Desktop/DSA4266/final_processed_results"

if [ $slicing = 1 ]; then
    python3 tools/tile_yolo.py \
    -source "/c/Users/Dell/Desktop/DSA4266/test_images/" \
    -falsefolder $test_image_sliced_raw \
    -ext .jpg -size 640
fi

if [ $predicting = 1 ]; then
    python3 yolov3-master/detect.py \
    --weights $model_weights \
    --source $test_image_sliced_raw \
    --project $test_image_sliced_predict \
    --save-txt
fi

if [ $pasting = 1 ]; then
    python3 tools/image_concate.py \
    --image_path $test_image_sliced_predict \
    --image_prefix "20210903_095054" "20210729_132515" \
    --final_results_path $final_results_stored
fi

if [ $image_count = 1 ]; then 
    python3 tools/image_count.py \
    --image_labels_path $test_image_sliced_predict_labels \
    --image_label_prefix "20210903_095054" "20210729_132515" \
    --final_results_path $final_results_stored

fi 