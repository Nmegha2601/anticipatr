import torch.utils.data
import torchvision

def build_dataset(args, mode):
    if args.dataset == 'ek':
        from datasets.ek import build_ek_anticipation
        return build_ek_anticipation(args=args, mode=mode)

    elif args.dataset == 'bf':
        from datasets.bf import build_bf_anticipation
        return build_bf_anticipation(args=args, mode=mode)

