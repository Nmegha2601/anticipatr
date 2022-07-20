"""
  Builds a dataloader class for snippet-level anticipation task
"""

import numpy as np
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
from .baseds_snippetprediction import SequenceDatasetLongFuture

def build_bf_pretraining(args,mode,override_modality=None):
        path_to_features = "{}/{}/{}/features/".format(args.root, args.dataset, args.anticipation) + "i3d_feats.pkl"
        label_type = 'verb'
        path_to_csv = '{}/{}/{}/split/{}_S{}.csv'.format(args.root, args.dataset, args.anticipation, mode, args.split, label_type)
        manyshot_anns = {'verb':'data/bf/longfuture/annotations/bf_verbs.csv'} 
        pretraining_train_vids = "pretraining_data/bf/train_videos.txt"
        pretraining_val_vids = "pretraining_data/bf/val_videos.txt"

        train_timestamps = [float(t) for t in args.train_timestamps.split(',')]
        val_timestamps = [float(t) for t in args.val_timestamps.split(',')]

        kwargs = { 
            'feature_file': path_to_features,
            'ann_file': path_to_csv,
            'label_type': args.label_type,
            'test_mode': False if mode == 'train' else True,
            'task': args.task,
            'fps': args.fps, 
            'dset': args.dataset,
            'train_vid_list': pretraining_train_vids,
            'val_vid_list': pretraining_val_vids,
            'num_verbs': 48,
            'num_nouns': 1,
            'num_actions': 1,
            'train_many_shot': args.train_many_shot,
            'manyshot_annotations': manyshot_anns,     
            'pretraining_task': args.pretraining_task,
            'num_future_labels': args.num_future_labels
        }   

        dataset = SequenceDatasetLongFuture(**kwargs)

        return dataset 


