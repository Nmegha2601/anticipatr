import os
import argparse
import random
import numpy as np
import time
from pathlib import Path
import json
import datetime
import pickle
import torch
from torch.utils.data import DataLoader
import datasets
import utils.misc as utils
from datasets import build_dataset
from models import build_model
from engine import train_one_epoch, evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,default="bf")
parser.add_argument('--root',type=str,help='Path to data root directory')
parser.add_argument('--num_nouns',type=int,default=1)
parser.add_argument('--num_verbs',type=int,default=48)
parser.add_argument('--num_actions',type=int,default=1)
parser.add_argument('--task',type=str,default='anticipation',choices=['anticipation','recognition'])
parser.add_argument('--anticipation',type=str,default='longfuture',choices=['nearfuture','longfuture'])
parser.add_argument('--pretraining_task',type=str,default='snippet_longfuture_anticipation')
parser.add_argument('--fps',type=int,default=60)
parser.add_argument('--label_type',type=str,default='verb',choices=['verb','noun','action'])
parser.add_argument('--action_repr',type=str,default='actionset',choices=['actionset'])
parser.add_argument('--train_many_shot',action='store_true',default=False,help='training with many shot verbs')
parser.add_argument('--split',type=int,default=1)
parser.add_argument('--train_timestamps',type=str,default='0.2,0.3,0.4,0.5,0.6,0.7,0.8')
parser.add_argument('--val_timestamps',type=str,default='0.25,0.5,0.75')

# model parameters
parser.add_argument('--model',type=str,default='antr')
parser.add_argument('--matcher_type',type=str,default='greedy', choices=['hungarian','greedy'])
parser.add_argument('--num_queries',type=int,default=10)
parser.add_argument('--num_pos_embed_dict',type=int,default=50000)
parser.add_argument('--dim_latent',type=int,default=128)
parser.add_argument('--hidden_dim',type=int,default=256)
parser.add_argument('--position_embedding',type=str,default='sine')
parser.add_argument('--num_decoder_embedding',type=int,default=10000)
parser.add_argument('--position_type',type=str,default='index',choices=['time','index'])
parser.add_argument('--dropout',type=float,default=0.1,help='transformer droput')
parser.add_argument('--nheads',type=int,default=8)
parser.add_argument('--dim_feedforward',type=int,default=2048)
parser.add_argument('--encoder',type=str,default='parallel')
parser.add_argument('--decoder',type=str,default='parallel')
parser.add_argument('--enc_layers',type=int,default=2)
parser.add_argument('--dec_layers',type=int,default=2)
parser.add_argument('--pretrained_enc_layers',type=int,default=2)
parser.add_argument('--pretrained_dec_layers',type=int,default=2)
parser.add_argument('--snippet_window',type=int,default=16)
parser.add_argument('--pretrained_path',type=str,default='')
parser.add_argument('--pre_norm',action='store_true')
parser.add_argument('--aux_loss',action='store_true')
parser.add_argument('--cuda',action='store_true',help='gpu mode')
parser.add_argument('--eval',action='store_true',help='evaluation mode')
parser.add_argument('--norm_type',type=str,choices=['gn','bn'],default='bn',help="normalization type")
parser.add_argument('--activation',type=str,default='leaky_relu',help="transformer activation type")

parser.add_argument('--set_cost_class',type=float,default=1,help='transformer droput')
parser.add_argument('--set_cost_segment',type=float,default=5,help='transformer droput')
parser.add_argument('--set_cost_siou',type=float,default=3,help='transformer droput')
parser.add_argument('--loss_coef_segment',type=float,default=5,help='transformer droput')
parser.add_argument('--loss_coef_siou',type=float,default=3,help='transformer droput')
parser.add_argument('--eos_coef',type=float,default=0.1,help='transformer droput')

# * Training
parser.add_argument('--resume',type=str,default='',help='resume from a checkpoint')
parser.add_argument('--save_checkpoint_every',type=int,default=1000,help='checkpoint saving frequency')
parser.add_argument('--evaluate_every',type=int,default=5,help='checkpoint saving frequency')
parser.add_argument('--evaluate_every_epoch',type=int,default=5,help='checkpoint saving frequency')
parser.add_argument('--num_workers',type=int,default=0,help='number of workers')
parser.add_argument('--batch_size',type=int,default=2,help='batch_size')
parser.add_argument('--epochs',type=int,default=10,help='number of epochs')
parser.add_argument('--step_size',type=int,default=64,help='number of steps before backpropagation')
parser.add_argument('--start_epoch',type=int,default=0,help='starting epoch')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_joiner', default=0, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_drop', default=100, type=int)
parser.add_argument('--clip_max_norm', default=1, type=float,help='gradient clipping max norm')
parser.add_argument('--output_dir', type=str,default='./experiments/checkpoints/',help='path to save intermediate checkpoints')

# * Distributed Training
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--device', default='cuda',help='device to use for training / testing')


args = parser.parse_args()
print(args)

def main(args):
    bz = args.batch_size
    lr = args.lr

    if args.cuda:
        if torch.cuda.device_count() >= 1:
            utils.init_distributed_mode(args)
        device = torch.device(args.device) 
    else:
        device = torch.device('cpu')
   
    # fix the seed for reproducibility
    if args.cuda:
        seed = args.seed + utils.get_rank()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # datasets build
    dataset_train = build_dataset(args=args, mode="train")
    dataset_test = build_dataset(args=args, mode="val")

    if args.cuda and args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train,shuffle=True)
        sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # set up model
    model, criterion = build_model(args)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    if args.cuda and args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
            model_with_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # set up model training
    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if "joiner" not in n and p.requires_grad]},
        {"params": [p for n, p in model_without_ddp.named_parameters() if "joiner" in n and p.requires_grad], "lr": args.lr_joiner,},]


    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # output and checkpoints directory
    checkpoint_dir = args.output_dir
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass
 
    if args.resume:
        checkpoint = Path(args.resume)
        assert checkpoint.exists()

        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1


    print("Start Training")
    start_time = time.time() 
    optimizer.zero_grad()
    for epoch in range(args.start_epoch, args.epochs):
        if args.cuda and args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(epoch, args.clip_max_norm, model, criterion, data_loader_train, optimizer, lr_scheduler, device)

        if args.output_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_paths = [checkpoint_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_every == 0:
                checkpoint_paths.append(checkpoint_dir / f'checkpoint{epoch:05}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'args': args,}, checkpoint_path) 
        
        # evaluation
        if epoch % args.evaluate_every_epoch == 0:
            test_stats = evaluate(epoch, model, criterion, data_loader_test, args.dataset, args.evaluate_every, device)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()},'epoch': epoch, 'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},'epoch': epoch, 'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (checkpoint_dir / 'log.json').open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        lr_scheduler.step()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == "__main__":
    main(args)






                                                    
