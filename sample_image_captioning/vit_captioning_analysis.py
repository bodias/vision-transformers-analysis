from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from transformers import pipeline
import torch
import numpy as np
from PIL import Image
import cv2
import os

os.environ["WANDB_DISABLED"] = "true"

class VITCaptioningModel():
    def __init__(self, output_encoder_attentions=True, output_encoder_hidden_states=True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"model running on {self.device}")
        self.model, self.feature_extractor, self.tokenizer = self.model_initialization(output_encoder_attentions, output_encoder_hidden_states)

    def model_initialization(self, output_encoder_attentions=True, output_encoder_hidden_states=True):
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", output_attentions=True)
        model.config.encoder.output_attentions = output_encoder_attentions
        model.config.encoder.output_hidden_states = output_encoder_hidden_states

        feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        model.to(self.device)

        ## only used for caption generation. not needed for regualr forward pass that gets the
        ## attention maps
        # max_length = 16
        # num_beams = 4
        # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "output_attentions": 'true'}

        return model, feature_extractor, tokenizer

    #requires more inputs
    def forward_pass(self, img: Image):
        # text preprocessing step
        def tokenization_fn(captions, max_target_length):
            """Run tokenization on captions."""
            labels = self.tokenizer(captions, 
                            padding="max_length", 
                            max_length=max_target_length,return_tensors="pt").input_ids

            return labels

        pixel_values = self.feature_extractor(images=[img], return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(self.device)
        ## Can we use without this input???
        ## it does not seem to affect the output of the attention maps. 
        labels = tokenization_fn("", 128)
        labels = labels.to(self.device)
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs   

    def generate_caption(self,img: Image):
        #TODO : move these variable to a better place
        # max_length = 16
        # num_beams = 4
        # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "output_attentions": 'true'}

        # pixel_values = self.feature_extractor(images=[img], return_tensors='pt').pixel_values
        # pixel_values = pixel_values.to(self.device)

        # output_ids = self.model.generate(pixel_values, **gen_kwargs)

        # preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # preds = [pred.strip() for pred in preds]

        image_captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        return image_captioner(img)

    
    def get_all_attention_maps(self, attentions, renorm_weights=True, output_as_matrix=True):
        
        att_map = torch.stack(attentions).squeeze(1)

        # Average the attention weights across all heads.
        att_map = torch.mean(att_map, dim=1)
        mean_att_map = att_map

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.        
        if renorm_weights:
            residual_att = torch.eye(mean_att_map.size(1))
            aug_mean_map = mean_att_map + residual_att.to(self.device)
            aug_mean_map = aug_mean_map / aug_mean_map.sum(dim=-1).unsqueeze(-1)
            mean_att_map = aug_mean_map

        return mean_att_map

    def get_joint_attention_map(self, attentions, output_as_matrix=True):
        # preprocess attention maps
        aug_att_mat = self.get_all_attention_maps(attentions, renorm_weights=True)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions = joint_attentions.to(self.device)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        # Last layer attention map with all joint attentions
        return joint_attentions[-1]
    

def display_att_map(att_map, img_size:tuple, grid_size=14):
    # show CLS token against all other tokens except itself
    display_att_layer = att_map[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
    display_att_layer = cv2.resize(display_att_layer / display_att_layer.max(), img_size)

    return display_att_layer
    
def is_patch_within_mask(original_img_mask, patch_coord, mask_threshold=.70):
    patch_with_mask = original_img_mask[patch_coord[0]:patch_coord[1], patch_coord[2]:patch_coord[3],:]
    # previously was dividing by 768, why? 768 is the embbedding size...
    perc_mask_pixels = len(patch_with_mask[patch_with_mask==255]) / len(patch_with_mask.flatten())

    return perc_mask_pixels > mask_threshold, patch_with_mask

def find_original_img_patch(vit_patch:int, original_img, grid_size:int=14, patch_size:int=16):
#     h_p, w_p = vit_patch
#     projection = original_img[h_p * patch_size:(h_p * patch_size)+patch_size, w_p * patch_size:(w_p * patch_size)+patch_size]
    col_p = vit_patch // grid_size
    row_p = vit_patch - (col_p * grid_size)
    y = row_p * patch_size
    width = patch_size
    x = col_p * patch_size
    height = patch_size
    projection = original_img[x:x+width, y:y+height]
    return projection, (x, x+width, y, y+height)

def xy_coord_token(token, grid_size=14):
    y = token // grid_size
    x = token - (y * grid_size)
    return x, y

def find_mask_tokens(img, mask, mask_threshold, n_tokens = 196):
    """
    img and mask have to be resized to work (224,224)
    """
    img_patches = []
    mask_patches = np.zeros((n_tokens), dtype="bool")
    mask_tokens = []
    for patch_i in range(n_tokens):
        img_patch, coord = find_original_img_patch(patch_i, img)
        mask_patches[patch_i] = is_patch_within_mask(mask, coord, mask_threshold)[0]
        if mask_patches[patch_i]:
            mask_tokens.append(patch_i)
        img_patches.append(img_patch)
        
    return mask_tokens, mask_patches, img_patches    
    

    