#!/bin/bash
# conda activate thesis-tf
start=$(date +%s)
# Task 1, person + accessory
## python feature_extraction.py task1_30-04_overlap train2017 1 ../filtered_datasets/train2017/task1_person_accessory_data.json
## python feature_extraction.py task1_30-04_overlap val2017 1 ../filtered_datasets/val2017/task1_person_accessory_data.json
# python feature_extraction.py task1_30-04_clean train2017 1 ../filtered_datasets/train2017/task1_person_accessory_data_no_overlap.json
# python feature_extraction.py task1_30-04_clean val2017 1 ../filtered_datasets/val2017/task1_person_accessory_data_no_overlap.json
# Task 2, dining table + objects
## python feature_extraction.py task2_30-04_overlap train2017 67 ../filtered_datasets/train2017/task2_diningtable_objects_data.json
## python feature_extraction.py task2_30-04_overlap val2017 67 ../filtered_datasets/val2017/task2_diningtable_objects_data.json
# python feature_extraction.py task2_30-04_clean train2017 67 ../filtered_datasets/train2017/task2_diningtable_objects_data_no_overlap.json
# python feature_extraction.py task2_30-04_clean val2017 67 ../filtered_datasets/val2017/task2_diningtable_objects_data_no_overlap.json
# task 4 and 5 combined, dining table (cake, pizza) and person (cake, pizza)
# python feature_extraction.py task4_5_30-04_overlap train2017 67 ../filtered_datasets/train2017/task4_dining_table_data.json 1 ../filtered_datasets/train2017/task5_person_data.json
# python feature_extraction.py task4_5_30-04_overlap val2017 67 ../filtered_datasets/val2017/task4_dining_table_data.json 1 ../filtered_datasets/val2017/task5_person_data.json
python feature_extraction.py task4_5_30-04_clean train2017 67 ../filtered_datasets/train2017/task4_dining_table_data_no_overlap.json 1 ../filtered_datasets/train2017/task5_person_data_no_overlap.json
python feature_extraction.py task4_5_30-04_clean val2017 67 ../filtered_datasets/val2017/task4_dining_table_data_no_overlap.json 1 ../filtered_datasets/val2017/task5_person_data_no_overlap.json
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
exec bash

