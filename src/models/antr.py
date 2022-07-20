import torch
import torch.nn.functional as F
from torch import nn

from .transformer import build_transformer
from .joiner import build_joiner
from .matcher import build_matcher
import snippet_models
import numpy as np
from utils.misc import accuracy, get_world_size, get_rank,is_dist_avail_and_initialized
from utils import segment_utils as segment_utils


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



class ANTR(nn.Module):
    def __init__(self, joiner, transformer, output_type, dim_feedforward, num_classes, num_queries, num_decoder_embedding, aux_loss = True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of action classes
            num_queries: number of anticipation queries, ie, decoder ouputs. This is the maximal number of actions the model can predict given a video.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.joiner = joiner
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.output_type = output_type
        self.num_queries = num_queries 
        self.query_embed = nn.Embedding(num_queries, hidden_dim) 
        self.query_time_embed = nn.Linear(hidden_dim + 1, hidden_dim)
             
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.segments_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.input_proj = nn.Conv1d(2048, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss

    def forward(self, samples, mask, targets,tgt_mask=None):
        """ The forward expects two inputs:
               - samples: batched videos features, of shape [batch_size x 2048 x T]
               - mask: a binary mask of shape [batch_size x T], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (start_time, end_time). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        assert mask is not None

        sample_positions = torch.empty_like(mask).to(samples.device) ## for positional encodings
        src, pos = self.joiner(samples,mask,sample_positions)
        input = self.input_proj(src)
        b, l, c = input.size()

        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
        nq = query_pos.size(0)
        prediction_times = torch.stack([t['prediction_duration'] for t in targets], axis=0).squeeze(1).repeat(1, nq, 1).permute(1,2,0)
        query_and_prediction_times = torch.cat([query_pos, prediction_times], axis=2)
        decoder_pos = self.query_time_embed(query_and_prediction_times.reshape(b * nq, self.hidden_dim + 1)).reshape(b,nq,-1).permute(1,0,2)
        hs = self.transformer(input,src, mask, tgt_mask, decoder_pos,pos)[0]
        outputs_class = self.class_embed(hs)
        outputs_segments = self.segments_embed(hs)
        outputs_segments = F.relu(outputs_segments) + 0.1
        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_segments[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_segments)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_segments):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b} for a, b in zip(outputs_class[:-1], outputs_segments[:-1])]


class CriterionGreedyMatcher(nn.Module):
    """ This class computes the loss for ANTICIPATR.
    The process happens in two steps:
        1) we compute greedy assignment between ground truth segments and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and segment)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of action categories, omitting the special no-action category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-action category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.matcher = matcher
        self.losses = losses
        self.eos_coef=eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src,_) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_segments]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the segments, the L1 regression loss and the IoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [num_segments, 2]
        """
        assert 'pred_segments' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx].squeeze(1)
        target_segments = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze(1)
        loss_segment = F.l1_loss(src_segments, target_segments, reduction='none')
        losses = {}
        losses['loss_segment'] = loss_segment.sum()/num_segments
        loss_siou = 1 - torch.diag(segment_utils.generalized_segment_iou(src_segments,target_segments))
        losses['loss_siou'] = loss_siou.sum()/num_segments

        return losses

    def get_unrolled_timeline(self, outputs, targets):
        src_logits = F.softmax(outputs['pred_logits'],dim=2)
        b,q,c = src_logits.size()
        src_segments = outputs['pred_segments']
        scale_factor = torch.cat([t['prediction_duration'].unsqueeze(0) for t in targets]).repeat(1,2)
        src_segments_scaled = src_segments * scale_factor[:,None,:]

        fps = targets[0]['fps']
        out_logits = torch.zeros(b,int(torch.round(torch.max(torch.cat([t['prediction_duration'] for t in targets])))),self.num_classes+1).to(src_logits.device)   
        for bidx in range(b):
            for sidx in range(len(src_segments_scaled[bidx])):
                s = max(int(src_segments_scaled[bidx][sidx][0]), 0)
                e = min(int(src_segments_scaled[bidx][sidx][1]), out_logits.size(1))
                for tidx in range(s,e):
                    out_logits[bidx,tidx,:] = torch.max(out_logits[bidx,tidx,:], src_logits[bidx][sidx])

        output_classes_onehot = torch.tensor(F.one_hot(torch.argmax(out_logits[:,:,:-1],dim=2),num_classes=self.num_classes),dtype=torch.float32)

        target_classes = torch.zeros(b,out_logits.size(1),c-1).to(src_logits.device)
        for idx, t in enumerate(targets):
            target_classes[idx,:t['labels_onehot'].size(0),:] = t['labels_onehot']

        return output_classes_onehot, target_classes

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_segments):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty segments
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-action" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def get_mAP(self, pred, labels, label_mask):
        mAPs = dict()
        pred = torch.clip(pred.sum(1), min=0.0,max=1.0)
        labels = torch.clip(labels.sum(1), min=0.0,max=1.0)
        for i in range(label_mask.shape[1]):
            if torch.sum(label_mask[0][i]) > 0:
                pred_i = pred[:, label_mask[0][i]].squeeze(1)
                labels_i = labels[:, label_mask[0][i]].squeeze(1)
                mAPs['AP_{}'.format(i)] = torch.cat((pred_i.detach().cpu(), labels_i.detach().cpu()),1)
        
        return mAPs   

    def get_accuracy(self,pred,labels, outputs, targets):
        acc = dict()
        for i in range(pred.shape[0]):
          acc['acc_{}_{}'.format(int(targets[i]['ratio_idx']*100), int(targets[i]['prediction_idx']*100))] = torch.cat((pred[i].detach().cpu(), labels[i].detach().cpu()),1)
            
        return acc   


    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'segments': self.loss_segments,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        all_indices = self.matcher(outputs_without_aux, targets)
        indices = [idx[:,(idx[1,:] + 1).nonzero(as_tuple=False)] for idx in all_indices]
        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t["labels"]) for t in targets)
        num_segments = torch.as_tensor([num_segments], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_segments))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        pred,labels = self.get_unrolled_timeline(outputs, targets)
        losses.update(self.get_mAP(pred,labels, targets[0]['label_mask']))
        losses.update(self.get_accuracy(pred, labels, outputs, targets))

        return losses

def build(args):
    joiner = build_joiner(args)
 
    transformer = build_transformer(args)

    if args.label_type == 'verb':
        num_classes = args.num_verbs

    if args.label_type == 'noun':
        num_classes = args.num_nouns

    if args.label_type == 'action':
        num_classes = args.num_actions
        
    num_queries = args.num_queries

    model = ANTR(
      joiner,
      transformer,
      dim_feedforward=args.dim_feedforward,        
      output_type=args.action_repr,       
      num_classes=num_classes,
      num_queries=num_queries,
      num_decoder_embedding=args.num_decoder_embedding,
      aux_loss=args.aux_loss,
      
    )

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'segments', 'cardinality']

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_segment': args.loss_coef_segment, 'loss_siou': args.loss_coef_siou}
    criterion = CriterionGreedyMatcher(num_classes, matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)

    print(model)
    return model, criterion





