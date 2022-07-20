import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import utils.segment_utils as segment_utils

class GreedyMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_action. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-actions).
    """

    def __init__(self):
        """Creates the matcher

        """
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_segments": Tensor of dim [batch_size, num_queries, 2] with the predicted segment timestamps

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_segments] (where num_target_segments is the number of ground-truth
                           actions in the target) containing the class labels
                 "segmentes": Tensor of dim [num_target_segments, 2] containing the target segment timestamps

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_segmentes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_segment = outputs["pred_segments"].flatten(0, 1)  # [batch_size * num_queries, 2]

        scale_factor = torch.stack([t["prediction_duration"] for t in targets], dim=0)
        out_segment_scaled = out_segment * scale_factor.unsqueeze(1).repeat(1,num_queries,1).flatten(0,1).repeat(1,2)           

        tgt_segment = torch.cat([v["segments"] for v in targets])
        tgt_segment_scaled = torch.cat([v["segments"] * v['prediction_duration'] for v in targets])
        
        indices = []
        for i in range(bs):
            targets_i = targets[i]['segments'] * targets[i]['prediction_duration']
            targets_i = targets_i[torch.sort(targets_i[:,1]-targets_i[:,0], descending=True)[1]]
            preds_i = outputs['pred_segments'][i] * scale_factor[i][0]
            tgt_i = []
            p_i = []
            seen_pidx = []
            for tidx, tgt in enumerate(targets_i):
                sorted_iou = torch.sort(segment_utils.generalized_segment_iou(tgt.unsqueeze(0), preds_i), descending=True,dim=0,stable=True)[1]
                for s in sorted_iou:
                    if s not in seen_pidx:
                        pidx = s
                        seen_pidx.append(s)
                        break
                tgt_i.append(tidx)
                p_i.append(pidx)
            unseen_pidx = [p for p in range(num_queries) if p not in seen_pidx]
            for up_idx in unseen_pidx:
                p_i.append(up_idx)
                tgt_i.append(-1)
            indices.append(torch.cat((torch.as_tensor(p_i,dtype=torch.int64).unsqueeze(0),torch.as_tensor(tgt_i,dtype=torch.int64).unsqueeze(0)),dim=0))

        sizes = [len(v["segments"]) for v in targets]
        return [torch.as_tensor(i, dtype=torch.int64) for i in indices]

def build_matcher(args):
    return GreedyMatcher()



