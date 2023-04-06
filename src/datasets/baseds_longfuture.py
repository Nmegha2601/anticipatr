import bisect
import copy
import os 
import os.path as osp
import random
from functools import partial
import itertools
import numpy as np
import pickle as pkl
import collections
from collections import Sequence
import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from datasets import ds_utils

class DatasetSegmentRecord(object):
    def __init__(self, row, clip_range=None):
        self._data = row
        self.clip_range = clip_range

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        return [int(x) for x in self._data[3:]]


    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1

    @property
    def clip_start_frame(self):
        return int(self._data[1]) if self.clip_range is None else int(self.clip_range[0])

    @property
    def clip_end_frame(self):
        return int(self._data[2]) if self.clip_range is None else int(self.clip_range[1])


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def get_many_shot(fin):
    with open(fin, "r") as f:
        lines = f.readlines()[1:]
        classes = [int(line.split(',')[0]) for line in lines]
    
    return classes


class SequenceDatasetLongFuture(Dataset):
    def __init__(self, feature_file, ann_file, label_type, test_mode, task, fps, dset, action_repr, prediction_type, train_timestamps, val_timestamps, num_verbs, num_nouns, num_actions, train_many_shot=False, manyshot_annotations={}, **kwargs):
        self.feature_file = feature_file
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.label = label_type
        self.task = task
        self.dset = dset
        self.action_repr = action_repr
        self.prediction_type = prediction_type
        self.train_many_shot = train_many_shot
        self.train_timestamps = train_timestamps
        self.val_timestamps = val_timestamps
        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        self.num_actions = num_actions
        self.train_prediction_interval = 10 ## time in seconds ; used only in training
        self.fps = fps
        
        if train_many_shot:
            manyshot_verbs = sorted(get_many_shot(manyshot_annotations['verb']))
            manyshot_nouns = sorted(get_many_shot(manyshot_annotations['noun']))
            self.num_verbs, self.num_nouns = len(manyshot_verbs), len(manyshot_nouns)
            self.manyshot_verbs, self.manyshot_nouns = manyshot_verbs, manyshot_nouns
        else:
            manyshot_nouns, manyshot_verbs = [],[]

        records = [DatasetSegmentRecord(x.strip().split('\t')) for x in open(self.ann_file)]
        if self.dset in ['ek','egtea']:
            int_counts = [(record.label[0], record.label[1]) for record in records]
            int_counts = collections.Counter(int_counts).items()
            int_counts = sorted(int_counts, key=lambda x: -x[1])[0:self.num_actions]
            self.int_to_idx = {interact:idx for idx, (interact, count) in enumerate(int_counts)}

        else:
            self.int_to_idx = {}

        if prediction_type=='time_independent':
            self.data = self.load_annotations_anticipation_time_independent(ann_file)
        elif prediction_type=='time_conditioned':
            self.data = self.load_annotations_anticipation_time_conditioned(ann_file)

        if train_many_shot:
            for record in self.data:
                record.verbs = [manyshot_verbs.index(x) for x in record.verbs if x in manyshot_verbs]
                record.nouns = [manyshot_nouns.index(x) for x in record.nouns if x in manyshot_nouns]
        
        # Only a few nouns/ints will actually have gt positives
        # Pass these as part of the batch to evaluate mAP
        # Don't know how to pass these in the config
        eval_ints = set()
        if self.dset in ['ek','egtea']:
            for record in self.data:
                eval_ints |= set(record.ints)
            eval_set = torch.zeros(1, self.num_actions)
            eval_set[0, list(eval_ints)] = 1
            self.eval_ints = eval_set.byte()
        else:
            self.eval_ints = torch.zeros(1, self.num_actions).byte()

        eval_nouns = set()
        if self.dset in ['ek','egtea']:
            for record in self.data:
                eval_nouns |= set(record.nouns)
            if not train_many_shot:
                eval_set = torch.zeros(1, self.num_nouns)
                eval_set[0, list(eval_nouns)] = 1
                self.eval_nouns = eval_set.byte()
            else:
                eval_set = torch.zeros(3, self.num_nouns)
                eval_set[0, list(eval_nouns)] = 1
                manyshot = eval_nouns & set(manyshot_nouns)
                rareshot = eval_nouns - set(manyshot_nouns)
                eval_set[1, list(manyshot)] = 1
                eval_set[2, list(rareshot)] = 1
                self.eval_nouns = eval_set.byte()

        else:
            self.eval_nouns = torch.zeros(1, self.num_actions).byte()


        eval_verbs = set()
        for record in self.data:
            eval_verbs |= set(record.verbs)

        if not train_many_shot:
            eval_set = torch.zeros(1, self.num_verbs)
            eval_set[0, list(eval_verbs)] = 1
        else:
            eval_set = torch.zeros(3, self.num_verbs)
            eval_set[0, list(eval_verbs)] = 1
            manyshot = eval_verbs & set(manyshot_verbs)
            rareshot = eval_verbs - set(manyshot_verbs)
            eval_set[1, list(manyshot)] = 1
            eval_set[2, list(rareshot)] = 1
        self.eval_verbs = eval_set.byte()

        self.prepare = RecordAnticipationData(self.action_repr, self.prediction_type, self.feature_file, self.dset, self.num_nouns, self.num_verbs, self.num_actions, self.int_to_idx, self.fps, self.label, self.eval_verbs, self.eval_nouns, self.eval_ints)


    def load_annotations_anticipation_time_independent(self, ann_file):
        vid_lengths = open(self.ann_file.replace('.csv', '_nframes.csv')).read().strip().split('\n')
        vid_lengths = [line.split('\t') for line in vid_lengths]
        vid_lengths = {k:int(v) for k,v in vid_lengths}

        records = [DatasetSegmentRecord(x.strip().split('\t')) for x in open(ann_file)]

        records_by_vid = collections.defaultdict(list)
        for record in records:
            record.uid = '%s_%s_%s'%(record.path, record.start_frame, record.end_frame)
            records_by_vid[record.path].append(record)

        records = []

        for vid in records_by_vid:
            vrecords = sorted(records_by_vid[vid], key=lambda record: record.end_frame)
            length = vid_lengths[vid]

            if self.test_mode:
                timestamps = self.val_timestamps
            else:
                timestamps = self.train_timestamps                           # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
       
            timestamps = [int(frac*length) for frac in timestamps]
            for i, t in enumerate(timestamps):
                past_records = [record for record in vrecords if record.end_frame<=t]
                future_records = [record for record in vrecords if record.start_frame>t]
                if len(past_records)<3 or len(future_records)<3:
                    continue

                record = DatasetSegmentRecord([vid, 0, t, -1, -1])
                if self.dset in ['ek','egtea']:
                    record.instances = [dict(segment=[record.start_frame,record.end_frame], verb=record.label[0], noun=record.label[1], action=self.int_to_idx[(record.label[0],record.label[1])]) for record in future_records if (record.label[0],record.label[1]) in self.int_to_idx]
                    record.nouns = sorted(set([record.label[1] for record in future_records]))
                    record.ints = sorted(set([self.int_to_idx[(record.label[0], record.label[1])] for record in future_records if (record.label[0], record.label[1]) in self.int_to_idx]))
                record.verbs =sorted(set([record.label[0] for record in future_records]))
                record.fps = self.fps
                record.ratio_idx = i
                record.prediction_idx = 1
                record.duration = length
                record.prediction_duration = length - t
                record.observation_duration = t
                records.append(record)
                       
        print(self.dset, ": time-independent anticipation", len(records))
        return records

    def load_annotations_anticipation_time_conditioned(self, ann_file):
        vid_lengths = open(self.ann_file.replace('.csv', '_nframes.csv')).read().strip().split('\n')
        vid_lengths = [line.split('\t') for line in vid_lengths]
        vid_lengths = {k:int(v) for k,v in vid_lengths}

        records = [DatasetSegmentRecord(x.strip().split('\t')) for x in open(ann_file)]
        records_by_vid = collections.defaultdict(list)
        for record in records:
            record.uid = '%s_%s_%s'%(record.path, record.start_frame, record.end_frame)
            records_by_vid[record.path].append(record)

        records = []

        for vid in records_by_vid:
            vrecords = sorted(records_by_vid[vid], key=lambda record: record.end_frame)
            length = vid_lengths[vid]

            if self.test_mode:
                timestamps = self.val_timestamps
                unseen_timestamps = [0.1, 0.2, 0.3, 0.4, 0.5]
            else:
                timestamps = self.train_timestamps                           # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                unseen_timestamps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

      
            seen_timestamps = [int(frac*length) for frac in timestamps]    
            for i, t in enumerate(seen_timestamps):
                past_records = [record for record in vrecords if record.end_frame<=t]
                prediction_timestamps = [int(frac*(length - t)) + t for frac in unseen_timestamps]    
                # prediction_timestamps = [min(t,length-t) for pt in prediction_timestamps] 
                for j, pred_t in enumerate(prediction_timestamps):
                    future_records = [record for record in vrecords if record.start_frame>t and record.end_frame<=pred_t]
                    if len(past_records)<3 or len(future_records)<3:
                        continue
                    record = DatasetSegmentRecord([vid, 0, t, -1, -1])
                    record.instances = [dict(segment=[record.start_frame,record.end_frame], verb=record.label[0]) for record in future_records]

                    record.verbs =sorted(set([record.label[0] for record in future_records]))
                    record.fps = self.fps
                    record.ratio_idx = timestamps[i]
                    record.prediction_idx = unseen_timestamps[j]
                    record.duration = length
                    record.prediction_duration = pred_t - t
                    record.observation_duration = t
                    records.append(record)

        print(self.dset,": time-conditioned anticipation", len(records))
        return records

    def get_ann_info(self, idx):
        return {
            'path': self.data[idx].path,
            'num_frames': self.data[idx].num_frames,
            'label': self.data[idx].label
        }
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vrecord = self.data[idx]
        inputs, targets = self.prepare(vrecord)
        return inputs, targets


class RecordAnticipationData(object):
    def __init__(self, action_repr, prediction_type, feature_file, dset, num_nouns, num_verbs, num_actions, int_to_idx, fps, label_type, eval_verbs, eval_nouns, eval_actions): 
        self.action_repr = action_repr
        self.prediction_type = prediction_type
        self.feature_file = feature_file
        self.dset = dset
        self.num_nouns = num_nouns
        self.num_verbs = num_verbs
        self.num_actions = num_actions
        self.int_to_idx = int_to_idx
        self.fps = fps
        self.label_type = label_type
        self.eval_verbs = eval_verbs
        self.eval_nouns = eval_nouns
        self.eval_actions = eval_actions
       

        with open(feature_file,'rb') as f:
            self.feature_data = pkl.load(f)

    def __call__(self, vrecord):
        ## features of past records
        vidname = vrecord.path
        duration = vrecord.duration
        features = []
        observation_positions = []
        for idx in range(vrecord.start_frame-31,vrecord.end_frame+31):
            if idx in self.feature_data[vidname].keys():
                # set fps to choose the sampling rate (TODO: set as argument)
                fps = 1
                if idx% fps ==0:
                   features.append(self.feature_data[vidname][idx])
                   observation_positions.append(idx)
        features = torch.tensor(features,dtype=torch.float32).permute(1,0)           
        observation_positions = torch.tensor(observation_positions,dtype=torch.float32)
        video_id = ds_utils.getVideoId(self.dset, vidname)            
        

        ## output representation
        set_targets = {}
        set_targets['video_id'] = torch.tensor(video_id)

        if self.label_type == 'action':
            label = torch.zeros(self.num_actions) 
            label[vrecord.ints] = 1
            set_targets['labels_onehot'] = to_tensor(label)
            set_targets['labels'] = torch.tensor([instance['action'] for instance in vrecord.instances])
            num_classes = self.num_actions            
            set_targets['label_mask'] = to_tensor(self.eval_actions)

        elif self.label_type == 'verb':
            label = torch.zeros(self.num_verbs) 
            label[vrecord.verbs] = 1
            set_targets['labels_onehot'] = to_tensor(label)
            set_targets['labels'] = torch.tensor([instance['verb'] for instance in vrecord.instances])
            num_classes = self.num_verbs            
            set_targets['label_mask'] = to_tensor(self.eval_verbs)

        elif self.label_type == 'noun':
            label = torch.zeros(self.num_nouns) 
            label[vrecord.nouns] = 1
            set_targets['labels_onehot'] = to_tensor(label)
            set_targets['labels'] = torch.tensor([instance['nouns'] for instance in vrecord.instances]) 
            num_classes = self.num_nouns            
            set_targets['label_mask'] = to_tensor(self.eval_nouns)


        set_targets['segments'] = [(np.asarray(instance['segment']) - vrecord.observation_duration)/vrecord.prediction_duration for instance in vrecord.instances] 
        set_targets['segments'] = torch.tensor(set_targets['segments'],dtype=torch.float32) 
        set_targets['labels_onehot'] = torch.tensor(set_targets['labels_onehot'], dtype=torch.float32)
        set_targets['duration'] = torch.tensor([vrecord.duration/self.fps],dtype=torch.float32)
        set_targets['prediction_duration'] = torch.tensor([vrecord.prediction_duration],dtype=torch.float32)
        set_targets['observation_duration'] = torch.tensor([(vrecord.end_frame - vrecord.start_frame)],dtype=torch.float32)
        set_targets['ratio_idx'] = torch.tensor([vrecord.ratio_idx],dtype=torch.float32)   
        set_targets['prediction_idx'] = torch.tensor([vrecord.prediction_idx],dtype=torch.float32)    
        set_targets['observation_positions'] = observation_positions
        set_targets['fps'] = torch.tensor([vrecord.fps],dtype=torch.float32)
        return features, set_targets
                            
