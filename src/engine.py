import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os,sys
import copy
import numpy as np
import math
from typing import Iterable
import time

import utils.misc as utils
import datasets


from metrics.longfuture_metrics import AnticipationEvaluator

def train_one_epoch(epoch, max_norm, model, criterion, data_loader, optimizer, scheduler, device):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    step = 0 

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step += 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        tgt_mask = None
        outputs = model(samples.tensors, samples.mask, targets, tgt_mask)

        losses = criterion(outputs, targets)
        loss_dict = {k:v for k,v in losses.items() if 'loss' in k}
        weight_dict = criterion.weight_dict
        loss_value = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        loss_value.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if 'AP' not in k}

    print("Train epoch:", epoch, "Averaged stats:", train_stats)
    return train_stats


def evaluate(epoch, model, criterion, data_loader, dataset, evaluate_every, device):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Test: [{}]'.format(epoch)
    print_freq = 50
    step = 0 

    predictions = {}
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        step += 1
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        tgt_mask = None
        outputs = model(samples.tensors, samples.mask,targets, tgt_mask)

        losses = criterion(outputs, targets)
        losses_metric = {k:v for k,v in losses.items() if 'AP' in k or 'acc' in k}
        losses_metric = [{k:v[i] for k,v in losses_metric.items()} for i in range(samples.tensors.size(0))]
        loss_dict = {k:v for k,v in losses.items() if 'loss' in k}
        weight_dict = criterion.weight_dict
        loss_value = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        metric_logger.update(loss=losses_reduced_scaled.item(), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)

        res = {datasets.ds_utils.getVideoName(dataset, target['video_id'].tolist()): output for target, output in zip(targets,losses_metric)}

        predictions.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
  
    ######For mAP calculation need to gather all data###########

    all_predictions = utils.all_gather(predictions)
    stats = {}

    if epoch % evaluate_every == 0:
      evaluator = AnticipationEvaluator(dataset)
      test_stats = evaluator.evaluate(all_predictions)


    print("Test epoch:", epoch, "Averaged test stats:",  test_stats)
    return test_stats
    
