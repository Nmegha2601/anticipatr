"""
  Constructs a dataloader for Epic-Kitchens-55 for the task of long term action anticipation.
"""
import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd

from .baseds_longfuture import SequenceDatasetLongFuture
#verbs, nouns,action: 125,3522,3806
#train_many_shot --verb,noun,action: 26,32,250

def build_ek_anticipation(args,mode,override_modality=None):
    path_to_features = "{}/{}/{}/features/".format(args.root, args.dataset, args.anticipation) + "{}_lfb_s30_{}.pkl".format(mode,'verb')
    label_type = '' if args.label_type == 'action' else args.label_type
    path_to_csv = '{}/{}/{}/split/{}_S{}.csv'.format(args.root, args.dataset, args.anticipation, mode, args.split, label_type)
    manyshot_anns = {'verb':'data/ek/longfuture/annotations/EPIC_many_shot_verbs.csv', 'noun':'data/ek/longfuture/annotations/EPIC_many_shot_nouns.csv'} 
    
    train_timestamps = [float(t) for t in args.train_timestamps.split(',')]
    
    timestamps = '0.25,0.5,0.75'
    val_timestamps = [float(t) for t in timestamps.split(',')]
    
    kwargs = {
        'feature_file': path_to_features,
        'ann_file': path_to_csv,
        'label_type': args.label_type,
        'test_mode': False if mode == 'train' else True,
        'task': args.task,
        'fps': args.fps, 
        'dset': args.dataset,
        'action_repr': args.action_repr,
        'prediction_type': 'time_independent',
        'train_timestamps': train_timestamps,
        'val_timestamps': val_timestamps,
        'num_verbs': args.num_verbs ,
        'num_nouns': args.num_nouns,
        'num_actions': args.num_actions,
        'train_many_shot': args.train_many_shot,
        'manyshot_annotations': manyshot_anns
    }
    
    dataset = SequenceDatasetLongFuture(**kwargs)
    
    return dataset 
