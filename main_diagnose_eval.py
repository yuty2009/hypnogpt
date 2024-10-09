
import os
import csv
import time
import json
import tqdm
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from timm.optim.optim_factory import param_groups_weight_decay

import os, sys; sys.path.append(os.path.dirname(__file__)+"./")
import torchutils as utils
from engine import train_epoch, evaluate, sensitivity_specificity
from engine import SeqClassificationDataset, SleepSequenceCollator
from models.gpt_longseq import GPTLongSeqTransformer


sleep_datasets = {
    'cap_sleepedf' : {
        'data_dir' : 'e:/eegdata/sleep/cap_sleepedf/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/cap_sleepedf/output/',
    },
    'isruc' : {
        'data_dir' : 'e:/eegdata/sleep/isruc/',
        'output_dir' : 'e:/eegdata/sleep/isruc/output/',
    },
}

parser = argparse.ArgumentParser(description='Train and evaluate a GPT model')
parser.add_argument('-D', '--dataset', default='isruc', metavar='PATH',
                help='dataset used')
parser.add_argument('-a', '--arch', metavar='ARCH', default='gpt',
                help='model architecture (default: gpt)')
parser.add_argument('--seg_seqlen', default=60, type=int, metavar='N',
                    help='maximum acceptable sequence length (default: 30)')
parser.add_argument('--embed_dim', default=48, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=3, type=int, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('--num_heads', default=6, type=int, metavar='N',
                    help='number of heads for multi-head attention (default: 6)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--folds', default=10, type=int, metavar='N',
                    help='number of folds cross-valiation (default: 20)')
parser.add_argument('--start-fold', default=0, type=int, metavar='N',
                    help='manual fold number (useful on restarts)')
parser.add_argument('--splits', default='', type=str, metavar='PATH',
                    help='path to cross-validation splits file (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--schedule', default='cos', type=str,
                    choices=['cos', 'step'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[0.6, 0.8], nargs='*', type=float,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-s', '--save-freq', default=10, type=int,
                metavar='N', help='save frequency (default: 100)')
parser.add_argument('-e', '--evaluate', default=False, action='store_true',
                    help='evaluate on the test dataset')


def main():

    args = parser.parse_args()

    args.data_dir = sleep_datasets[args.dataset]['data_dir']
    args.output_dir = sleep_datasets[args.dataset]['output_dir']
    args.pretrained = "output/gpt_cap_pretrained/best.pth.tar"

    output_prefix = f"{args.arch}_eval"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(1243)

    # Data loading code
    print(f"=> loading dataset from '{args.data_dir}'")
    ann_f = open(os.path.join(args.data_dir, 'annotations.txt'), newline='')
    reader = csv.reader(ann_f)
    seqdata = list(reader)
    # seqdata = crop_sleep_period(seqdata)
    sub_f = open(os.path.join(args.data_dir, 'subject_labels.txt'), newline='')
    reader = csv.reader(sub_f)
    labeldata = np.array(list(reader)).flatten()
    labeldata = [int(lb) for lb in labeldata]
    labeldata = np.asarray(labeldata)
    labeldata[labeldata > 0] = 1 # 0: normal, 1: abnormal
    
    print('Data for %d subjects has been loaded' % len(labeldata))
    num_subjects = len(labeldata)

    args.vocab_size = 6     # 5 sleep stages + padding token (5)
    
    args.writer = None
    with open(args.output_dir + "/args.json", 'w') as fid:
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        json.dump(args.__dict__, fid, indent=2, default=default)
    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log"))

    test_dataset = SeqClassificationDataset(seqdata, labeldata)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=SleepSequenceCollator(args.seg_seqlen),
    )

    model = GPTLongSeqTransformer(
        num_classes = 2,
        vocab_size = args.vocab_size,
        seg_seqlen = args.seg_seqlen,
        embed_dim = args.embed_dim,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        pretrained = args.pretrained,
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")
    criterion = nn.CrossEntropyLoss().to(args.device)

    model = model.to(args.device)
    
    test_loss, test_accu, y_trues, y_probs = evaluate(test_loader, model, criterion, 0, args)
    
    y_preds = np.argmax(y_probs, axis=-1)
    test_cm = confusion_matrix(y_true=y_trues, y_pred=y_preds)
    sensi, speci = sensitivity_specificity(test_cm)
    print(f"Test accu: {test_accu:.3f}, loss: {test_loss:.3f}, "
          f"Sensitivity: {sensi:.3f}, Specificity: {speci:.3f}")
    print(test_cm)


if __name__ == "__main__":

    main()
