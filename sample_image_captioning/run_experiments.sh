#!/bin/bash
# conda activate thesis-tf
start=$(date +%s)
## task 1 experiments with two types of object mask and non overlapping dataset
# python training_decoding_nn.py task1_exp_30-04_mask-3_person_accessory_clean train2017/task1_30-04_clean/features-mask-3-main_thr-0-sec_thr-0/ val2017/task1_30-04_clean/features-mask-3-main_thr-0-sec_thr-0/
# python training_decoding_nn.py task1_exp_30-04_mask-4_person_accessory_clean train2017/task1_30-04_clean/features-mask-4-main_thr-0-sec_thr-0/ val2017/task1_30-04_clean/features-mask-4-main_thr-0-sec_thr-0/
## task 2 experiments with two types of object mask and non overlapping dataset
python training_decoding_nn.py task2_exp_30-04_mask-3_diningtable_objects_clean train2017/task2_30-04_clean/features-mask-3-main_thr-0-sec_thr-0/ val2017/task2_30-04_clean/features-mask-3-main_thr-0-sec_thr-0/
python training_decoding_nn.py task2_exp_30-04_mask-4_diningtable_objects_clean train2017/task2_30-04_clean/features-mask-4-main_thr-0-sec_thr-0/ val2017/task2_30-04_clean/features-mask-4-main_thr-0-sec_thr-0/
## task with two main objects and non overlapping dataset
python training_decoding_nn.py task4_5_exp_30-04_mask-3_dining_person_food_clean train2017/task4_5_30-04_clean/features-mask-3-main_thr-0-sec_thr-0/ val2017/task4_5_30-04_clean/features-mask-3-main_thr-0-sec_thr-0/
python training_decoding_nn.py task4_5_exp_30-04_mask-4_dining_person_food_clean train2017/task4_5_30-04_clean/features-mask-4-main_thr-0-sec_thr-0/ val2017/task4_5_30-04_clean/features-mask-4-main_thr-0-sec_thr-0/
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
exec bash
