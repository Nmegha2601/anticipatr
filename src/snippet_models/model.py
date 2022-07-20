import torch
import torch.nn.functional as F
from torch import nn

from .transformer import build_transformer
from .joiner import build_joiner

import numpy as np
from utils.misc import accuracy, get_world_size, get_rank,is_dist_avail_and_initialized

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x): 
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class EncoderSnippetLongfutureAnticipation(nn.Module):
    def __init__(self, joiner, transformer, dim_feedforward, num_classes, num_queries, aux_loss = True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of action classes
            num_queries: number of action queries, ie decoder outputs.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.joiner = joiner
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes)

        self.input_proj = nn.Conv1d(2048, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss

    def forward(self, samples, mask, targets=None, tgt_mask=None):
        """ The forward expects two inputs:
               - samples.tensor: batched videos features, of shape [batch_size x 2048 x T]
               - samples.mask: a binary mask of shape [batch_size x T], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (start_time, end_time). These values are normalized in [0, 1],
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        assert mask is not None
        sample_positions=torch.empty_like(mask)
        src, pos = self.joiner(samples,mask,sample_positions)
        input = self.input_proj(src)
        hs = self.transformer(input,mask, tgt_mask, self.query_embed.weight,pos)[0]
        outputs_class = self.class_embed(hs)

        return outputs_class

class CriterionSnippetLongfutureAnticipation(nn.Module):
    def __init__(self, num_classes, weight_dict, eos_coef, losses,fps):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def get_mAP(self,pred,labels,label_mask):
        mAPs = dict()
    
        for i in range(label_mask.shape[0]):
            pred_i = pred[:, label_mask[i]]
            labels_i = labels[:, label_mask[i]]
            mAPs['mAP_{}'.format(i)] = torch.cat((pred_i.detach().cpu(), labels_i.detach().cpu()),1)
            if mAPs['mAP_{}'.format(i)].ndim == 1:
                mAPs['mAP_{}'.format(i)] = mAPs['mAP_{}'.format(i)].unsqueeze(0) 
        return mAPs   


    def loss_labels(self,outputs, targets,log=True):
        src_logits = torch.sigmoid(outputs['pred_logits'].mean(1))
        target_classes = torch.cat([t['labels'].unsqueeze(0) for t in targets])
        loss_ce = F.binary_cross_entropy(src_logits, target_classes,reduction='mean')
        losses = {'loss_ce': loss_ce}
        losses.update(self.get_mAP(src_logits, target_classes, targets[0]['label_mask']))

        return losses


    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'labels': self.loss_labels
        }

        assert loss in loss_map, f'{loss} loss not defined'
        return loss_map[loss](outputs,targets,**kwargs)


    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        if 'aux_outputs' in outputs:
            for i,aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    kwargs = {'log' : False}
                    l_dict = self.get_loss(loss,aux_outputs,targets,**kwargs)
                    l_dict = {k + f'_{i}': v for k,v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def replace_last_layer(model):
    model.class_embed = Identity()
    return model

def build(args):
    joiner = build_joiner(args)
    transformer = build_transformer(args)

    if args.label_type == 'verb':
        num_classes = args.num_verbs

    if args.label_type == 'noun':
        num_classes = args.num_nouns

    if args.label_type == 'action':
        num_classes = args.num_actions

    losses = ['labels']
    weight_dict = {'loss_ce':1}
    model = EncoderSnippetLongfutureAnticipation(
      joiner,
      transformer,
      dim_feedforward=args.dim_feedforward,
      num_classes=num_classes,
      num_queries=1,
      aux_loss=args.aux_loss,
    )

    criterion = CriterionSnippetLongfutureAnticipation(num_classes=num_classes, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,fps=args.fps)
    
    model = replace_last_layer(model)
    print(model)
    return model


                          
