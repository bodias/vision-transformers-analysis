#!/bin/bash
# conda activate thesis-tf3006-05_
start=$(date +%s)
# Task 1, person + accessory
## python feature_extraction.py task1_30-04_overlap train2017 1 ../filtered_datasets/train2017/task1_person_accessory_data.json
## python feature_extraction.py task1_30-04_overlap val2017 1 ../filtered_datasets/val2017/task1_person_accessory_data.json
# python feature_extraction.py task1_30-04_L5-8_clean train2017 1 ../filtered_datasets/train2017/task1_person_accessory_data_no_overlap.json
# python feature_extraction.py task1_30-04_L5-8_clean val2017 1 ../filtered_datasets/val2017/task1_person_accessory_data_no_overlap.json
# python feature_extraction.py task1_30-04_L5-8_clean_caption train2017 1 ../filtered_datasets/train2017/task1_person_accessory_data_no_overlap_caption.json
# python feature_extraction.py task1_30-04_L5-8_clean_caption val2017 1 ../filtered_datasets/val2017/task1_person_accessory_data_no_overlap_caption.json
# Task 2, dining table + objects
## python feature_extraction.py task2_30-04_overlap train2017 67 ../filtered_datasets/train2017/task2_diningtable_objects_data.json
## python feature_extraction.py task2_30-04_overlap val2017 67 ../filtered_datasets/val2017/task2_diningtable_objects_data.json
# python feature_extraction.py task2_30-04_L5-8_clean train2017 67 ../filtered_datasets/train2017/task2_diningtable_objects_data_no_overlap.json
# python feature_extraction.py task2_30-04_L5-8_clean val2017 67 ../filtered_datasets/val2017/task2_diningtable_objects_data_no_overlap.json
# python feature_extraction.py task2_30-04_L5-8_clean_caption train2017 67 ../filtered_datasets/train2017/task2_diningtable_objects_data_no_overlap_caption.json
# python feature_extraction.py task2_30-04_L5-8_clean_caption val2017 67 ../filtered_datasets/val2017/task2_diningtable_objects_data_no_overlap_caption.json
# task 4 and 5 combined, dining table (cake, pizza) and person (cake, pizza)
## python feature_extraction.py task4_5_30-04_overlap train2017 67 ../filtered_datasets/train2017/task4_dining_table_data.json 1 ../filtered_datasets/train2017/task5_person_data.json
## python feature_extraction.py task4_5_30-04_overlap val2017 67 ../filtered_datasets/val2017/task4_dining_table_data.json 1 ../filtered_datasets/val2017/task5_person_data.json
# python feature_extraction.py task4_5_30-04_L5-8_clean train2017 67 ../filtered_datasets/train2017/task4_dining_table_data_no_overlap.json 1 ../filtered_datasets/train2017/task5_person_data_no_overlap.json
# python feature_extraction.py task4_5_30-04_L5-8_clean val2017 67 ../filtered_datasets/val2017/task4_dining_table_data_no_overlap.json 1 ../filtered_datasets/val2017/task5_person_data_no_overlap.json
# python feature_extraction.py task4_5_30-04_L5-8_clean_caption train2017 67 ../filtered_datasets/train2017/task4_dining_table_data_no_overlap_caption.json 1 ../filtered_datasets/train2017/task5_person_data_no_overlap_caption.json
# python feature_extraction.py task4_5_30-04_L5-8_clean_caption val2017 67 ../filtered_datasets/val2017/task4_dining_table_data_no_overlap_caption.json 1 ../filtered_datasets/val2017/task5_person_data_no_overlap_caption.json

# Full training using all layers to show training plot
# python feature_extraction.py task4_5_30-04_L5-8_clean_caption val2017 67 ../filtered_datasets/val2017/task4_dining_table_data_no_overlap_caption.json 1 ../filtered_datasets/val2017/task5_person_data_no_overlap_caption.json 2,3,4,5,6,7,8,9,10,11


# Task 1, person + accessory
python feature_extraction.py task1_06-05_clean train2017 1,2,3,4,5,6,7,8,9,10,11 1 ../filtered_datasets/train2017/task1_person_accessory_data_no_overlap.json 
python feature_extraction.py task1_06-05_clean val2017 1,2,3,4,5,6,7,8,9,10,11 1 ../filtered_datasets/val2017/task1_person_accessory_data_no_overlap.json 
python feature_extraction.py task1_06-05_clean_caption train2017 1,2,3,4,5,6,7,8,9,10,11 1 ../filtered_datasets/train2017/task1_person_accessory_data_no_overlap_caption.json 
python feature_extraction.py task1_06-05_clean_caption val2017 1,2,3,4,5,6,7,8,9,10,11 1 ../filtered_datasets/val2017/task1_person_accessory_data_no_overlap_caption.json 
# Task 2, dining table + objects
python feature_extraction.py task2_06-05_clean train2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/train2017/task2_diningtable_objects_data_no_overlap.json 
python feature_extraction.py task2_06-05_clean val2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/val2017/task2_diningtable_objects_data_no_overlap.json 
python feature_extraction.py task2_06-05_clean_caption train2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/train2017/task2_diningtable_objects_data_no_overlap_caption.json 
python feature_extraction.py task2_06-05_clean_caption val2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/val2017/task2_diningtable_objects_data_no_overlap_caption.json 
# task 4 and 5 combined, dining table (cake, pizza) and person (cake, pizza)
python feature_extraction.py task4_5_06-05_clean train2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/train2017/task4_dining_table_data_no_overlap.json 1 ../filtered_datasets/train2017/task5_person_data_no_overlap.json
python feature_extraction.py task4_5_06-05_clean val2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/val2017/task4_dining_table_data_no_overlap.json 1 ../filtered_datasets/val2017/task5_person_data_no_overlap.json
python feature_extraction.py task4_5_05-05_clean_caption train2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/train2017/task4_dining_table_data_no_overlap_caption.json 1 ../filtered_datasets/train2017/task5_person_data_no_overlap_caption.json 
python feature_extraction.py task4_5_05-05_clean_caption val2017 1,2,3,4,5,6,7,8,9,10,11 67 ../filtered_datasets/val2017/task4_dining_table_data_no_overlap_caption.json 1 ../filtered_datasets/val2017/task5_person_data_no_overlap_caption.json 

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
exec bash

