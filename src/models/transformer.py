"""
Transformer class.

Code inspired by torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * decoder handles multiple encoders
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import os, sys
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from snippet_models import build_snippet_model

class TransformerMultipleEncoder(nn.Module):

    def __init__(self, snippet_model, d_model=512, nhead=8, num_encoder_layers=6,num_pretrained_layers=4,snippet_window=32,
                 num_decoder_layers=6, encoding='parallel', decoding='parallel', dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False,
                 return_intermediate_dec=False,pretrained_path=''):
        super().__init__()

        self.encoding = encoding
        self.decoding = decoding
        self.d_model = d_model
        self.nhead = nhead
        self.snippet_window = snippet_window

        ## construct video encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None 
        self.video_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.snippet_encoder = snippet_model
        for param in self.snippet_encoder.parameters():
            param.requires_grad = False

        # load pretrained snippet encoder
        if pretrained_path != '' and os.path.exists(pretrained_path):
            self.snippet_encoder.eval()
            model_dict = self.snippet_encoder.state_dict()
            pretrained_dict = torch.load(pretrained_path,map_location=torch.device('cpu'))['model']
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.snippet_encoder.load_state_dict(model_dict)

        ## construct decoder
        decoder_layer = TransformerMultipleEncoderDecoderLayer(d_model, nhead, dim_feedforward,
                                                               dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerMultipleEncoderDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, orig_src, src_mask, tgt_mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, t = src.shape
        src = src.permute(2, 0, 1)
        orig_src = orig_src.permute(2,0,1)
        pos_embed = pos_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        encoder_mask = None
        memory_snippet = None

        ## extracting snippet representations and handling overflow properly
        ## overflow needs to be handled as video length might not be a multiple
        ## of  the size of snippet length used in snippet encoder
        if t % self.snippet_window == 0:
            memory_snippet = self.snippet_encoder(orig_src.reshape(bs * (t//self.snippet_window), -1, self.snippet_window), 
                                                  mask=torch.zeros((bs * (t//self.snippet_window),self.snippet_window),dtype=torch.bool).to(orig_src.device)).reshape(bs,-1,t)
        else:
            overflow = t % self.snippet_window
            windows_length = t - (t % self.snippet_window)
            windows_memory = self.snippet_encoder(orig_src[:windows_length,:,:].reshape(bs * (windows_length//self.snippet_window), -1, self.snippet_window), 
                                                  mask=torch.zeros((bs * (windows_length//self.snippet_window), self.snippet_window),dtype=torch.bool).to(orig_src.device))
            overflow_memory = self.snippet_encoder(orig_src[-overflow:,:,:].reshape(bs, -1, overflow), mask=torch.zeros((bs,overflow),dtype=torch.bool).to(orig_src.device))
            memory_snippet = torch.cat((windows_memory.reshape(bs,-1,windows_length), overflow_memory.reshape(bs, -1, overflow)), dim=2)
        memory_video = self.video_encoder(src, mask=encoder_mask, src_key_padding_mask=src_mask, pos=pos_embed)
        memory_snippet = memory_snippet.reshape(t,bs,c)
        tgt_mask = None
        hs = self.decoder(tgt, memory_snippet, memory_video, tgt_mask, memory_key_padding_mask_1=src_mask, memory_key_padding_mask_2=src_mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory_video.permute(1, 2, 0)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerMultipleEncoderDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate


    def forward(self, tgt, memory_1, memory_2,
                tgt_mask: Optional[Tensor] = None,
                memory_mask_1: Optional[Tensor] = None,
                memory_mask_2: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask_1: Optional[Tensor] = None,
                memory_key_padding_mask_2: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        for layer in self.layers:
            output = layer(output, memory_1, memory_2, tgt_mask=tgt_mask,
                           memory_mask_1=memory_mask_1, memory_mask_2=memory_mask_2,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask_1=memory_key_padding_mask_1,
                           memory_key_padding_mask_2=memory_key_padding_mask_2,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerMultipleEncoderDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="leaky_relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos 

    def forward_post(self, tgt, memory_1, memory_2,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask_1: Optional[Tensor] = None,
                     memory_mask_2: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask_1: Optional[Tensor] = None,
                     memory_key_padding_mask_2: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn1(query=self.with_pos_embed(tgt, query_pos),
                                   key=memory_1,
                                   value=memory_1, attn_mask=memory_mask_1,
                                   key_padding_mask=memory_key_padding_mask_1)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn2(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory_2, pos),
                                   value=memory_2, attn_mask=memory_mask_2,
                                   key_padding_mask=memory_key_padding_mask_2)[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt 


    def forward(self, tgt, memory_1, memory_2,
                tgt_mask: Optional[Tensor] = None,
                memory_mask_1: Optional[Tensor] = None,
                memory_mask_2: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask_1: Optional[Tensor] = None,
                memory_key_padding_mask_2: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory_1, memory_2, tgt_mask, memory_mask_1, memory_mask_2,
                                 tgt_key_padding_mask, memory_key_padding_mask_1,
                                 memory_key_padding_mask_2, pos, query_pos)




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
     snippet_model = build_snippet_model(args)

     return TransformerMultipleEncoder(
       snippet_model,
       d_model=args.hidden_dim,
       dropout=args.dropout,
       nhead=args.nheads,
       dim_feedforward=args.dim_feedforward,
       num_encoder_layers=args.enc_layers,
       num_pretrained_layers=args.pretrained_enc_layers,
       num_decoder_layers=args.dec_layers,
       snippet_window=args.snippet_window,
       encoding=args.encoder,
       decoding=args.decoder,
       activation=args.activation,
       normalize_before=args.pre_norm,
       return_intermediate_dec=True,
       pretrained_path=args.pretrained_path, 
       #combination_mode=args.combination
     )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "elu":
        return F.elu
    if activation == "leaky_relu":
        return F.leaky_relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
