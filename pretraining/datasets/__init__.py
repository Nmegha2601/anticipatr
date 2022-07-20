import torch.utils.data
import torchvision

def build_dataset(args):
    if args.dataset == 'ek':
        from datasets.ek import build_ek_pretraining
        dataset_train = build_ek_pretraining(args,mode='train')
        dataset_val = build_ek_pretraining(args,mode='val')
        return dataset_train, dataset_val

    elif args.dataset == 'bf':
        from datasets.bf import build_bf_pretraining
        dataset_train = build_bf_pretraining(args,mode='train')
        dataset_val = build_bf_pretraining(args,mode='val')
        return dataset_train, dataset_val
