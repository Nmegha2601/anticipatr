"""
    Evaluator class for action anticipation benchmarks
"""
import math
import numpy as np
import torch
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore", category=UserWarning)
import sklearn.metrics as skmetrics	

class AnticipationEvaluator(object):
    def __init__(self,dataset):
        self.apmeter = OrderedDict()
        self.output = OrderedDict()
        if dataset in ['ek','egtea']:
            prediction_type = 'time_independent'
        elif dataset in ['bf','salads']:
            prediction_type = 'time_conditioned'
        self.prediction_type = prediction_type
        self.accmeter = OrderedDict()
        self.output['mAP_micro'] = []
        self.output['mAP_macro'] = []

    def get_AP_perclass(self, predictions):
        if isinstance(predictions,dict):
            predictions = [predictions]
        preds = {}
        preds['mAP'] = []
        targets = {}
        targets['mAP'] = []

        for p in predictions:
            for k,v in p.items():
              for k_ap,v_ap in v.items():
                 if 'mAP' in k_ap:
                   preds[k_ap].append(v_ap[:v_ap.size(0)//2].numpy())
                   targets[k_ap].append(v_ap[v_ap.size(0)//2:].numpy())
        for k_ap,v_ap in preds.items():      
            y_true = np.asarray([t for t in targets[k_ap]])
            y_pred = np.asarray([p for p in preds[k_ap]])
            if 'mAP' in k_ap:
                self.output['mAP_macro'].append(skmetrics.average_precision_score(np.asarray(targets[k_ap]), np.asarray(preds[k_ap]), average='macro'))
                self.output['mAP_micro'].append(skmetrics.average_precision_score(np.asarray(targets[k_ap]), np.asarray(preds[k_ap]), average='micro'))



    def get_accuracy_perclass(self, predictions):
        if isinstance(predictions,dict):
            predictions = [predictions]
        preds = {}
        targets = {}
        for p in predictions:
            for k,v in p.items():
              for k_ap,v_ap in v.items():
                if 'acc' in k_ap:
                    if k_ap not in self.accmeter:
                        preds[k_ap] = []
                        targets[k_ap] = []
                    if v_ap.ndim == 1:
                        v_ap = v_ap.unsqueeze(0)
                    preds[k_ap].append(v_ap[:,:v_ap.size(1)//2].numpy())
                    targets[k_ap].append(v_ap[:,v_ap.size(1)//2:].numpy())
        for k,v in predictions[0].items():
          for k_ap,v_ap in v.items():       
            if 'acc' in k_ap and v_ap.size(0) > 0:
                self.output[k_ap] = []
                preds[k_ap] = np.asarray(preds[k_ap])
                preds[k_ap] = preds[k_ap].reshape(-1, preds[k_ap].shape[-1])
                targets[k_ap] = np.asarray(targets[k_ap])
                targets[k_ap] = targets[k_ap].reshape(-1, targets[k_ap].shape[-1])
                for cls in range(targets[k_ap].shape[1]):
                    preds_logits = preds[k_ap][:,cls]
                    preds_i = np.zeros_like(preds_logits)
                    preds_i[np.argmax(preds_logits)] = 1
                    labels_i = targets[k_ap][:,cls]
                    self.output[k_ap].append((1-skmetrics.hamming_loss(labels_i,preds_i)) * 100)
            
    def evaluate(self,predictions):
        ## Epic-Kitchens-55 and EGTEA Gaze+ evaluation
        if self.prediction_type == 'time_independent':
            self.get_AP_perclass(predictions)
            metrics = {}
            for k,v in self.output.items():
                if 'mAP' in k:
                    metrics[k] = v
            return metrics
     
        ## Breakfast and 50Salads evaluation
        if self.prediction_type == 'time_conditioned':
            self.get_accuracy_perclass(predictions)
            metrics = {}
            for k,v in self.output.items():
                if 'acc' in k:
                    metrics[k] = np.mean(np.asarray(v))
            return metrics

