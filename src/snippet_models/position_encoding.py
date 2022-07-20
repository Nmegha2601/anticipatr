"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn



class PositionEmbeddingSineIndex(nn.Module):
    """
      Sinusoidal positional encodings based on sequence timestamps      
    """
    def __init__(self, num_pos_feats, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats 
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    def forward(self, x,mask):
        assert mask is not None
        not_mask = ~mask
        not_mask = not_mask.to(mask.device)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=2).flatten(2)
        pos = pos_x.permute(0, 2, 1)
        return pos

def build_position_encoding(args):
    N_steps = args.hidden_dim
    if args.position_embedding == 'sine' and args.position_type=='index':
        position_embedding = PositionEmbeddingSineIndex(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
