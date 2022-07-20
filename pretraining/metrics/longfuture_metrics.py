import math
import numpy as np
import torch
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore", category=UserWarning)
import sklearn.metrics as skmetrics	

class AnticipationEvaluator(object):
    """ The pretraining task is multilabel classification problem."""
    def __init__(self):
        self.apmeter = OrderedDict()
        self.output = OrderedDict()
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


            
    def evaluate(self,predictions):
        self.get_AP_perclass(predictions)
        metrics = {}
        for k,v in self.output.items():
            if 'mAP' in k:
                metrics[k] = v
        return metrics
     





                                                    
