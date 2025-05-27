import torch
from models import DSDBase
import torch.nn.functional as F

import torch
# from torchinfo import summary
from PIL import Image
import open_clip
# from inference_tool import get_preprocess
from open_clip import tokenizer




class DSD(DSDBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True, use_contrastive_loss=True, \
                         use_affil_loss=False)
        self.config = config
        self.create_and_load_pretrained(config)
        

    def create_and_load_pretrained(self, config):
        self.model, _ ,_ = open_clip.create_model_and_transforms("ViT-B/32")
        # self.model, _ ,_ = open_clip.create_model_and_transforms("ViT-B/32",pretrained="/home/zzl/open_clip_pytorch_model_b32.bin") #加载本地权重
               
                    
    def get_vis_emb(self, image, idx=None, label=None):
        img_emb = self.model.encode_image(image,normalize=True)
        return img_emb
        
    def get_txt_emb(self, text_ids, idx=None, label=None):
        txt_emb = self.model.encode_text(text_ids,normalize=True)
        return txt_emb
        

    def forward(self, image, text_ids, idx=None, label=None):
        img_emb = self.get_vis_emb(image)
        txt_emb=self.get_txt_emb(text_ids)
        loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
        return loss_contr, img_emb, txt_emb