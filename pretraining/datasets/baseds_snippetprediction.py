"""
Implementation of dataloader for snippet anticipation.

Code inspired by: https://github.com/facebookresearch/ego-topo
"""

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

class SequenceDatasetLongFuture(Dataset):
    def __init__(self, feature_file, ann_file, label_type, test_mode, task, fps, dset, train_vid_list, val_vid_list, num_verbs, num_nouns, num_actions, train_many_shot=False, manyshot_annotations={}, num_future_labels=-1,**kwargs):
        self.feature_file = feature_file
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.label = label_type
        self.task = task
        self.dset = dset
        self.train_many_shot = train_many_shot
        self.train_vid_list = train_vid_list
        self.val_vid_list = val_vid_list
        self.num_verbs = num_verbs
        self.num_nouns = num_nouns
        self.num_actions = num_actions
        self.fps = fps
        self.num_future_labels = num_future_labels

        with open(feature_file,'rb') as f:
            self.feature_data = pkl.load(f)

        if train_many_shot:
            manyshot_verbs = sorted(get_many_shot(manyshot_annotations['verb']))
            manyshot_nouns = sorted(get_many_shot(manyshot_annotations['noun']))
            if train_many_shot:
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
       
        self.data = self.load_longfuture_anticipation_annotations(ann_file)

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
            self.eval_ints = torch.zeros(1, self.num_actions).byte()
            self.eval_nouns = torch.zeros(1, self.num_nouns).byte()
        
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

        self.prepare = RecordSnippetLongfutureAnticipationData(self.feature_data, self.dset, self.num_nouns, self.num_verbs, self.num_actions, self.int_to_idx, self.fps, self.label, self.eval_verbs, self.eval_nouns, self.eval_ints,self.test_mode)

    def load_longfuture_anticipation_annotations(self, ann_file):
        print("Loading longfuture anticipation annotations")
        vid_lengths = open(self.ann_file.replace('.csv', '_nframes.csv')).read().strip().split('\n')
        vid_lengths = [line.split('\t') for line in vid_lengths]
        vid_lengths = {k:int(v) for k,v in vid_lengths}

        if self.test_mode:
            vidfile = self.val_vid_list
        else:
            vidfile = self.train_vid_list

        with open(vidfile,'rb') as f:
            vid_list = [line.rstrip().decode() for line in f]

        records = [DatasetSegmentRecord(x.strip().split('\t')) for x in open(ann_file)]
        records_by_vid = collections.defaultdict(list)
        for record in records:
            if self.dset=='ek':
                path = record.path.split('/')[-1]
            else:
                path = record.path
            if path in vid_list:
                record.uid = '%s_%s_%s'%(record.path, record.start_frame, record.end_frame)
                records_by_vid[record.path].append(record)

        records = []
        for vid in records_by_vid:
            vrecords = sorted(records_by_vid[vid], key=lambda record: record.end_frame)
            length = vid_lengths[vid]
            if vid not in self.feature_data:
                continue
            for segment_idx, segment_record in enumerate(vrecords[:-2]):
                record_length = segment_record.end_frame - segment_record.start_frame + 1
                if not any(x in self.feature_data[vid].keys() for x in  list(range(segment_record.start_frame, segment_record.end_frame+1))):
                    continue
                if self.dset in ['bf']:
                    invalid_verbs = [0]
                    if record_length <= 15:
                        continue
                    if segment_record.label[0] == 0:
                        continue
                if self.dset in ['salads']:
                    if record_length <= 15:
                        continue
                    invalid_verbs = [17, 18]
                    if segment_record.label[0] in [17,18]:
                        continue
                else:
                    if record_length <= 1:
                        continue
                    invalid_verbs = []
               # create snippet record: label has to be future labels
                future_records = [record for record in vrecords[segment_idx+1:-1]]
                record = segment_record
                record.verbs = sorted(set([frec.label[0] for frec in future_records]))
                if self.num_future_labels > 0:
                    record.verbs = record.verbs[:num_future_labels]
                if self.dset in ['ek', 'egtea']:
                    record.nouns = sorted(set([frec.label[1] for frec in future_records]))
                    record.ints = sorted(set([self.int_to_idx[(frec.label[0], frec.label[1])] for frec in future_records if (frec.label[0], frec.label[1]) in self.int_to_idx]))
                    if self.num_future_labels > 0:
                        record.nouns = record.nouns[:num_future_labels]
                        record.ints = record.ints[:num_future_labels]
                record.duration = record.end_frame - record.start_frame
                record.fps = self.fps
                records.append(record)
                #if len(records) == 8: return records
        
        print("Snippet based longfuture anticipation", len(records))
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

class RecordSnippetLongfutureAnticipationData(object):
    def __init__(self, feature_data, dset, num_nouns, num_verbs, num_actions, int_to_idx, fps, label_type, eval_verbs, eval_nouns, eval_actions,test_mode):
        self.feature_data = feature_data
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
        self.test_mode = test_mode

    def __call__(self, vrecord):
        ## features of past records
        vidname = vrecord.path
        duration = vrecord.duration
        features = []

        for idx in range(vrecord.start_frame,vrecord.end_frame+1):
            if idx in self.feature_data[vidname].keys():
                if self.dset in ['ek', 'egtea']:
                    features.append(torch.tensor(self.feature_data[vidname][idx]))
                if self.dset in ['bf','salads']:
                # process at 15fps
                    if idx%15==0:
                        features.append(torch.tensor(self.feature_data[vidname][idx]))

        features = torch.tensor(torch.stack(features),dtype=torch.float32).permute(1,0)
        
        video_id = ds_utils.getVideoId(self.dset, vidname)

        ## output representation
        set_targets = {}
        set_targets['video_id'] = torch.tensor(video_id)

        if self.label_type == 'action':
            label = torch.zeros(self.num_actions) 
            label[vrecord.ints] = 1 
            set_targets['labels'] = to_tensor(label)
            set_targets['label_mask'] = to_tensor(self.eval_actions)
   
        elif self.label_type == 'noun':
            label = torch.zeros(self.num_nouns) 
            label[vrecord.nouns] = 1 
            set_targets['labels'] = to_tensor(label)
            set_targets['label_mask'] = to_tensor(self.eval_nouns)

        elif self.label_type == 'verb':
            label = torch.zeros(self.num_verbs) 
            label[vrecord.verbs] = 1 
            set_targets['labels'] = to_tensor(label)
            set_targets['label_mask'] = to_tensor(self.eval_verbs)

        set_targets['fps'] = torch.tensor([vrecord.fps],dtype=torch.float32)
        set_targets['duration'] = torch.tensor([vrecord.duration/vrecord.fps],dtype=torch.float32)
        set_targets['start_frame'] = torch.tensor([vrecord.start_frame],dtype=torch.float32)
        set_targets['end_frame'] = torch.tensor([vrecord.end_frame],dtype=torch.float32)

        return features, set_targets

