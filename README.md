# vision-transformers-analysis

Current files and their usage:

1. **coco_dataset_filtering.ipynb**: from COCO dataset, filter images containing specific *category_id* (instance annotations) or words in *caption* (caption annotations). Main function is called `filter_complex_images`.
    * It shows stats about the dataset, how many instances per *category_id*, objects more likely to appear on the same image.
    * It transforms the data into a different format more suitable for consumption later.
    * 
2. **feature_extraction-captioning-VIT.ipynb**: Based on the filtered COCO datasets, extract the relevant tokens (features) that will be used later for classification (decoding). It outputs pickle files for each class pair (MAIN, SECOND) with layer activations and tokens. Each file is ~8GB so it's not doable to store them in a single file.
    * **Segmentation selection**: from all instances of an object, which one will be selected and then used to find the "object" token.
    * **Extract relevant token(s)**: from the selected objects, find the relevant tokens:
        * `max` and `min` activation per object (MAIN, SECOND)
        * Random token from each object
        * Max image token (whole image)
        * TBD: Consistent token
3. **decoding task**: With all features (activations of relevant tokens - THE token with "best" attention) and ground truth labels (object category), train a ANN to decode the activation from the token and predict the category of the object.