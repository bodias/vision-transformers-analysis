import os
import json
import pickle
import random
import copy
from PIL import Image, ImageDraw
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"

# uses common functions and not the ones declared in the beginning
# TODO: use only functions from library (vit_captioning_analysis) to avoid confusion
from vit_captioning_analysis import VITCaptioningModel
from vit_captioning_analysis import order_obj_instances
from vit_captioning_analysis import is_patch_within_mask, find_original_img_patch
from vit_captioning_analysis import xy_coord_token, find_mask_tokens

COCO_PATH = "../../../datasets/coco"
BATCH_SIZE = 6
grid_size = 14

# TODO: 
# * get secondary object based on proximity - closest object
# * limit objects closer to the main cluster of activation
#   or choose the main object from area in main cluster of activation 
def get_bbox_center(annotation):
     return np.array((annotation["bbox"][0] + int(annotation["bbox"][2]/2), 
             annotation["bbox"][1] + int(annotation["bbox"][3]/2)))

def get_largest_mask(annotation, obj, img_draw):
    # select main (largest) mask for the object class        
    objects_area = [(ann['id'], ann['area']) for ann in annotation['annotations']['annotations'] if ann['category_id']==obj and ann['iscrowd']==0]
    largest_obj = sorted(objects_area, key=lambda x: x[1], reverse=True)[0]
    largest_obj_ann = [ann for ann in annotation['annotations']['annotations'] if ann['id']==largest_obj[0]]
    largest_obj_ann = largest_obj_ann[0] #avoid more than one object with same area
    segmentation = [ann['segmentation'] for ann in annotation['annotations']['annotations'] if ann['id']==largest_obj_ann["id"]]
    for segment in segmentation[0]:
        try:
            img_draw.polygon(segment, fill ="#ffffff")
        except ValueError:
            return None
    return [largest_obj_ann]
    
def get_all_masks(annotation, obj, img_draw):
    segmentation = [ann['segmentation'] for ann in annotation['annotations']['annotations'] if ann['category_id']==obj]
    for segment in segmentation:
        if isinstance(segment, list):
            try:
                img_draw.polygon(segment[0], fill ="#ffffff")
            except ValueError:
                return None
    return None

def get_mask_with_attention(img, annotation, category_id, img_draw, layer_attention_maps):
    segm_with_att = order_obj_instances(img, category_id, annotation, layer_attention_maps)
    for segment in segm_with_att[0][0]['segmentation']:
        img_draw.polygon(segment, fill ="#ffffff")
    return segm_with_att[0]

def get_objects_mask(annotation, objects=[], option=1, attention_maps=None):
    sample_image = Image.open(os.path.join(COCO_PATH, 'train2017', annotation['image']['file_name'])).convert('RGB')    
    masks = []
    for idx, obj in enumerate(objects):
        pil_mask = np.zeros(shape=sample_image.size, dtype=np.uint8)
        pil_mask = Image.fromarray(np.moveaxis(pil_mask, 0, -1))
        img_draw = ImageDraw.Draw(pil_mask) 
        
        if option == 1:
            get_largest_mask(annotation, obj, img_draw)

        # merge all masks of the object class into one
        elif option == 2:
            get_all_masks(annotation, obj, img_draw)    
                    
        # get closest mask based on a reference mask (first object)
        # calculate the centroid and then find the other mask closest to this centroid
        # using euclidian distance
        elif option == 3:
            ## assume first object is the reference. Get the first object based on biggest mask
            if idx==0:
                ref_annotation = get_largest_mask(annotation, obj, img_draw)[0]
            else:
                ref_center = get_bbox_center(ref_annotation)
                distances = [(np.linalg.norm(ref_center - get_bbox_center(ann)), ann["id"]) for ann in annotation['annotations']['annotations'] if ann['category_id']==obj]
                closest_obj = distances[0]
                segmentation = [ann['segmentation'] for ann in annotation['annotations']['annotations'] if ann['id']==closest_obj[1]]
                for segment in segmentation[0]:
                    try:
                        img_draw.polygon(segment, fill ="#ffffff")
                    except ValueError:
                        return None
        
        elif option == 4:
            # this option produces 12 masks, one for each layer of the transformers encoder
            pil_mask = []
            for layer in range(12):
                att_layer_mask = np.zeros(shape=sample_image.size, dtype=np.uint8)
                att_layer_mask = Image.fromarray(np.moveaxis(att_layer_mask, 0, -1))
                img_draw = ImageDraw.Draw(att_layer_mask) 
                get_mask_with_attention(sample_image, 
                                        annotation, 
                                        obj, 
                                        img_draw, 
                                        attention_maps[layer][0, 1:])
                pil_mask.append(att_layer_mask)
        
        masks.append(pil_mask)        
        
    return masks

# TODO: Improve logic of consistent tokens to account for different variations of layers and not only the last N
def find_tokens_in_region(attention_map:np.array, 
                          img: np.array, 
                          mask: np.array, 
                          layers=[9,10,11], 
                          min_n=3, 
                          max_n=20,
                          mask_threshold=.75,
                          grid_size=14,
                          display_token_grid=False,
                          display_att_layers=False):
    tokens = {}
    all_top_n = []
    
    if display_token_grid:
        # show original mask
        plt.figure(figsize=(6, 6))
        plt.imshow(mask)
        plt.show()
        # set up the plot with 14x14 image patches
        fig, axs = plt.subplots(nrows=14, ncols=14, figsize=(6, 6))
    
    # find which tokens belong to the mask
    mask_tokens, mask_patches, img_patches = find_mask_tokens(img, mask, mask_threshold)
    # If there are not tokens related to the mask, it is probably
    # because the mask region inside the patch is too small
    # try to find mask tokens again with threshold=0.
    #    e.g.: if one pixel falls into the mask patch it means it is a mask token/patch
    if not mask_tokens:
        mask_tokens, mask_patches, img_patches = find_mask_tokens(img, mask, mask_threshold=0)
        if not mask_tokens:
            return None, None, None
        
    # for each n_layers in the attention_map
    # find all tokens
    for layer_no in layers: #enumerate(attention_map[n_layers:, :, :]):
        layer = attention_map[layer_no, :, :]
        tokens_layer_i = {}
        
        # MAX token whole image
        tokens_layer_i['max_image'] = np.argmax(layer[0, 1:])
        
        # set all background activation WITHIN background mask to -1
        img_att_map_mask = copy.deepcopy(layer[0, 1:])
        img_att_map_mask[~mask_patches] = 0
        
        # MIN MAX token within object/mask
        max_token_obj_layer_i = np.argmax(img_att_map_mask)
        # TODO: Improve logic to get min attention token considering only one region. 
        # when slicing np.array and using np.argmin the index returned is not relative to original array anymore
        # e.g.: np.argmin(img_map_fg[fg_mask_patch]) will return an index relative to the new array img_map_fg[fg_mask_patch]
        min_token_obj_layer_i = 0
        min_activation = np.max(img_att_map_mask)
        for token_i in mask_tokens:
            if img_att_map_mask[token_i] < min_activation:
                min_token_obj_layer_i = token_i
                min_activation = img_att_map_mask[token_i]

        tokens_layer_i['max_obj'] = max_token_obj_layer_i
        tokens_layer_i['min_obj'] = min_token_obj_layer_i
        
        # RANDOM token within object
        tokens_layer_i['random_obj'] = random.choice(mask_tokens)
        
        # Get top N activations of the layer
        # based on https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        top_n = np.argpartition(img_att_map_mask, -1 * max_n)[-1 * max_n:]
        top_n = top_n[np.argsort(img_att_map_mask[top_n])]
        top_n = top_n[::-1]
        all_top_n.append(top_n)  
        
        tokens[layer_no] = tokens_layer_i

        # show grid with all tokens found
        if display_token_grid:
            for patch_i, img_patch in enumerate(img_patches):
                row_p, col_p = xy_coord_token(patch_i)
                if mask_patches[patch_i]:
                    axs[col_p, row_p].imshow(img_patch)                       
                else:
                    axs[col_p, row_p].imshow(np.zeros_like(img_patch))
    #                 axs[col_p, row_p].set_title(mask_patches[patch_i])
                axs[col_p, row_p].axis('off')
            plt.show()
            
        if display_att_layers:
            plt.figure(figsize=(3, 3))
            plt.imshow(img_att_map_mask.reshape(grid_size, grid_size))
            plt.title(f"layer {layer_no}, shape {img_att_map_mask.shape} ")
            row_p, col_p = xy_coord_token(tokens_layer_i['min_obj'])
            plt.scatter(row_p, col_p, marker='_', c='blue')
            row_p, col_p = xy_coord_token(tokens_layer_i['max_obj'])
            plt.scatter(row_p, col_p, marker='+', c='red')
            row_p, col_p = xy_coord_token(tokens_layer_i['random_obj'])
            plt.scatter(row_p, col_p, marker='*', c='green')
            row_p, col_p = xy_coord_token(tokens_layer_i['max_image'])
            plt.scatter(row_p, col_p, marker='P', c='orange')
            plt.show()
    
    consistent_token = None
#     # FOR CONSISTENT TOKEN ACROSS SEVERAL LAYERS
#     # find if there's intersection with previous layer
#     for n in range(min_n, max_n, 2):
#         common_tokens = set()
#         for layer_no, layer_top_n in enumerate(all_top_n):
#             if layer_no==0:
#                 common_tokens = set(layer_top_n[:n])                
#             else:
#                 common_tokens = common_tokens.intersection(set(layer_top_n[:n]))
#                 if not common_tokens:
# #                     print(f"No common token with top {n}")
#                     break
#         if common_tokens:
# #             print(f"found common tokens with top {n}")
#             consistent_token = list(common_tokens)
#             if len(consistent_token)>1:
#                 # sum selected tokens across all layers to get top 1
#                 sums = dict(zip(consistent_token, [0] * len(consistent_token)))
#                 for layer in attention_map[n_layers:, :, :]:
#                     for token in consistent_token:
#                         sums[token] += layer[0, token].cpu().detach().numpy()
# #                 print(sums)
#                 consistent_token = max(sums, key=sums.get)
# #                 print(consistent_token)
#             else:
#                 consistent_token = consistent_token[0]
#             break    

    return tokens, consistent_token, all_top_n

def extract_tokens(img: Image.Image, mask:np.array, mean_att_map, layers:list, mask_threshold=.75, debug=False):
        
    # some regions of the image have intermediate values between 0 and 255
    # maybe resizing the image create these "intermediate" pixels.
    fg_mask_img = copy.deepcopy(mask)
    fg_mask_img[fg_mask_img==255] = 255
    fg_mask_img[fg_mask_img!=255] = 0
    fg_mask_img = fg_mask_img[:,:,np.newaxis]
    # get background mask by reversing the image mask
    bg_mask_img = copy.deepcopy(fg_mask_img)
    bg_mask_img[fg_mask_img==255] = 0
    bg_mask_img[fg_mask_img!=255] = 255
    
## Not using background tokens at the moment
#     bg_tokens, consistent_bg_token, _ = find_tokens_in_region(mean_att_map, np.array(img), bg_mask_img, mask_threshold=mask_threshold,
#                                                               display_token_grid=debug, display_att_layers=debug)
    fg_tokens, consistent_fg_token, _ = find_tokens_in_region(attention_map=mean_att_map, 
                                                              img=np.array(img), 
                                                              mask=fg_mask_img, 
                                                              layers=layers,
                                                              mask_threshold=mask_threshold,
                                                              display_token_grid=debug, 
                                                              display_att_layers=debug)
    
    return fg_tokens, consistent_fg_token

def load_dataset(filename):
    with open(filename, "r") as f:
        annotations = json.load(f)
    with open('../coco_category_id_map.pickle', 'rb') as handle:
        category_id_map = pickle.load(handle) 

    return annotations, category_id_map

def run_extraction(annotation_groups, object_mask_strategy, main_obj_mask_thr, second_obj_mask_thr):
    labels_map = {27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase'}
    labels_map = {44: 'bottle', 47: 'cup', 48: 'fork', 49: 'knife', 51: 'bowl', 62: 'chair'}
    feat_folder = f"features-mask-{str(object_mask_strategy)}-main_thr-{str(main_obj_mask_thr)}-sec_thr-{str(second_obj_mask_thr)}"

    for main_class, annotations_file in annotation_groups.items():
        for second_class, task_annotations in annotations_file.items():        
            features = {"image_id": [],
                "image_filename": [],
                "caption_filter": [], 
                "main_fg_tokens": [],
                "main_consistent_fg_token": [],
    #                 "main_bg_tokens": [],
    #                 "main_consistent_bg_token": [],                    
    #                 "second_bg_tokens": [],
    #                 "second_consistent_bg_token": [],
                "second_fg_tokens": [],
                "second_consistent_fg_token": [],
                "main_fg_tokens_act": [],
                "second_fg_tokens_act": [],
                "class": []}
            
            count_img_with_caption = 0
            count_img_no_caption  = 0
            warning_msg = True
            print(f"class ({main_class}, {second_class})")
            batch_input = []
            batch_ann = []
            for idx, ann in enumerate(tqdm(task_annotations)):
                # if ann['annotations']['second_object_in_caption']:
                if False:
                    count_img_with_caption += 1
                else:
                    count_img_no_caption += 1
                
                if count_img_no_caption >= 1000 and count_img_with_caption >= 1000:
                    print("Enough images (1000 with and without caption), stopping")
                    break
                
                # if count_img_no_caption >= 1000 and not ann['annotations']['second_object_in_caption']:
                if count_img_no_caption >= 1000 and not False:
                    if warning_msg:
                        print("Reached 1000 images without caption, collecting only images WITH caption now.")
                        warning_msg = False
                    continue
                # elif count_img_with_caption >= 1000 and ann['annotations']['second_object_in_caption']:
                elif count_img_with_caption >= 1000 and False:
                    continue                
                
                img = Image.open(os.path.join(COCO_PATH, 'train2017', ann['image']['file_name'])).convert('RGB')             
                batch_input.append(img)
                batch_ann.append(ann)
                
                if len(batch_input) == BATCH_SIZE:                
                    outputs = model.forward_pass(batch_input)                
                    layers_act = torch.stack(outputs.encoder_hidden_states).cpu().detach().numpy()
                    mean_att_map = model.get_all_attention_maps(torch.stack(outputs.encoder_attentions), renorm_weights=True)            
                    mean_att_map = mean_att_map.cpu().detach().numpy()
                    
                    for input_i in range(len(batch_input)):                    
                        masks = get_objects_mask(batch_ann[input_i], 
                                                 objects=[int(main_class), int(second_class)], 
                                                 option=object_mask_strategy, 
                                                 attention_maps=mean_att_map[:,input_i,:,:])
                        if masks is not None:
                            # attn based
                            if object_mask_strategy==4:
                                fg_mask_main = np.array(masks[0][11].resize((224,224)))
                                fg_mask_second = np.array(masks[1][11].resize((224,224)))
                            else:
                                fg_mask_main = np.array(masks[0].resize((224,224)))
                                fg_mask_second = np.array(masks[1].resize((224,224)))            
                            # MAIN OBJECT
                            main_fg_tokens, main_cons_fg_token = extract_tokens(img=img.resize((224,224)), 
                                                                                mask=fg_mask_main,
                                                                                mean_att_map=mean_att_map[:,input_i,:,:],
                                                                                layers=[3,4,9,10,11],
                                                                                mask_threshold=main_obj_mask_thr)
                            # SECOND OBJECT
                            sec_fg_tokens, sec_cons_fg_token = extract_tokens(img=img.resize((224,224)), 
                                                                            mask=fg_mask_second,
                                                                            mean_att_map=mean_att_map[:,input_i,:,:],
                                                                            layers=[3,4,9,10,11],
                                                                            mask_threshold=second_obj_mask_thr)
                            if main_fg_tokens is not None and sec_fg_tokens is not None:
                                features["image_id"].append(batch_ann[input_i]['image']['id'])
                                features["image_filename"].append(batch_ann[input_i]['image']['file_name'])
                                features["main_fg_tokens"].append(main_fg_tokens)
                                features["main_consistent_fg_token"].append(main_cons_fg_token)
                                features["second_fg_tokens"].append(sec_fg_tokens)
                                features["second_consistent_fg_token"].append(sec_cons_fg_token)
                                # skip the CLS token. Token number is from [0, 195]
    #                             layers_act = [layer_act.cpu().squeeze().detach().numpy()[1:,:] for layer_act in layer_activations]
                                token_act_per_layer = {}
                                for layer, tokens in main_fg_tokens.items():
                                    token_act_per_layer[layer] = {'min_obj': layers_act[layer][input_i][tokens['min_obj']+1],
                                                                'max_obj': layers_act[layer][input_i][tokens['max_obj']+1],
                                                                'random_obj': layers_act[layer][input_i][tokens['random_obj']+1],
                                                                'max_image': layers_act[layer][input_i][tokens['max_image']+1]}

                                features["main_fg_tokens_act"].append(token_act_per_layer)                    
                                token_act_per_layer = {}
                                for layer, tokens in sec_fg_tokens.items():
                                    token_act_per_layer[layer] = {'min_obj': layers_act[layer][input_i][tokens['min_obj']+1],
                                                                'max_obj': layers_act[layer][input_i][tokens['max_obj']+1],
                                                                'random_obj': layers_act[layer][input_i][tokens['random_obj']+1],
                                                                'max_image': layers_act[layer][input_i][tokens['max_image']+1]}                  
                                features["second_fg_tokens_act"].append(token_act_per_layer)                    
                                # features["caption_filter"].append(batch_ann[input_i]['annotations']['second_object_in_caption'])
                                features["caption_filter"].append(False)
                                features["class"].append(labels_map[int(second_class)])                    
                        else:
                            print(f"mask is None: {batch_ann[input_i]['image']['file_name']}")
                    batch_input = []
                    batch_ann = []
            feat_pd = pd.DataFrame(features)
            if len(feat_pd) > 0:
                os.makedirs(feat_folder, exist_ok=True)
                feat_pd.to_pickle(f"{feat_folder}/feat-tokens_act-{main_class}-{second_class}.pickle")

if __name__ == "__main__":
    model = VITCaptioningModel(output_encoder_attentions=True, output_encoder_hidden_states=True)
    # person_annotations, category_id_map = load_dataset("../task2_person_accessory_data_w_caption.json")
    diningtable_annotations, category_id_map = load_dataset("../task2_dining_objects_data.json")
    
    for mask_strategy in [3,4]:
        for obj_threshold in [0,5,20,40]:
            print(f"Generating features: mask:{str(mask_strategy)} main_thr: {str(obj_threshold)} sec_thr: {str(obj_threshold)}")
            # run_extraction(annotation_groups={1: person_annotations}, 
            #                object_mask_strategy=mask_strategy, 
            #                main_obj_mask_thr=obj_threshold, 
            #                second_obj_mask_thr=obj_threshold)
            run_extraction(annotation_groups={67: diningtable_annotations}, 
                           object_mask_strategy=mask_strategy, 
                           main_obj_mask_thr=obj_threshold, 
                           second_obj_mask_thr=obj_threshold)
     

    