import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from utils import read_json
from functools import partial
from torchvision import models
from torch.autograd import Variable

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply



class DSDBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=True,
                 use_contrastive_loss=False, use_affil_loss=False):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.temp = nn.Parameter(torch.ones([]) * config['temp1'])


    def get_vision_embeds(self, image):
        """
        vision_embeds: cls + patch embeds
        """
        return F.normalize(self.vision_proj(self.vision_encoder(image))[:, 0, :])

    def get_text_embeds(self, text_ids):
        """
        text_embeds: cls + sequence embeds
        """
        return F.normalize(self.text_proj(self.text_encoder(text_ids))[:, 0, :])


    def get_contr_loss(self, image_feat, text_feat, idx=None, label=None, config=None):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        # image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
            text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        else:
            image_feat_all = image_feat
            text_feat_all = text_feat


        logits = image_feat_all @ text_feat_all.t() / self.temp

        # print(logits)
        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)

            ## get matching matrix
            # idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            else:
                idx_all = idx

            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2
    
    


    