
import os
import csv
import time
import json
import tqdm
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torchutils as utils
from basemodel import BaseNN


parser = argparse.ArgumentParser(description='Train and evaluate a sequence classification model')
parser.add_argument('-D', '--dataset', default='shhs', metavar='PATH',
                help='dataset used')
parser.add_argument('--pretrained', 
                    default='',
                    metavar='PATH', help='path to pretrained model (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='basenn',
                help='model architecture (default: gpt)')
parser.add_argument('--max_seqlen', default=30, type=int, metavar='N',
                    help='maximum acceptable sequence length (default: 30)')
parser.add_argument('--embed_dim', default=64, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=1, type=int, metavar='N',
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
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
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


class SeqClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = []
        for seq_raw in sequences:
            seq = [int(s) for s in seq_raw]
            self.sequences.append(seq)
        self.labels = [int(lb) for lb in labels]
        self.labels = np.asarray(self.labels)
        self.labels[self.labels > 0] = 1 # 0: normal, 1: abnormal
        self.len = len(labels)

    def __getitem__(self, index):
        sequence = torch.LongTensor(self.sequences[index])
        label = self.labels[index]
        return sequence, label

    def __len__(self):
        return self.len
    

def collate_batch(batch):
    token_list, label_list, offsets = [], [], [0]
    for (_token, _label) in batch:
         label_list.append(_label)
         _token = torch.tensor(_token, dtype=torch.int64)
         token_list.append(_token)
         offsets.append(_token.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    token_list = torch.cat(token_list)
    return token_list, offsets, label_list


def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    
    y_trues, y_preds = [], []
    total_loss, total_num, total_correct = 0.0, 0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, offsets, target in data_bar:
        data = data.to(args.device)
        offsets = offsets.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data, offsets)
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_num += target.size(0)
        pred = torch.argmax(output, dim=-1)
        y_trues.append(target)
        y_preds.append(pred)

        total_correct += torch.sum((pred == target).float()).item()
        accu = 100 * total_correct / total_num

        info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
            epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
        info += "Accu: {:.2f}".format(accu) 
        data_bar.set_description(info)
    
    y_trues = torch.concatenate(y_trues).cpu().numpy()
    y_preds = torch.concatenate(y_preds).cpu().numpy()
    cm = confusion_matrix(y_trues, y_preds)
    
    return total_loss/len(data_loader), accu, cm


def evaluate(data_loader, model, criterion, epoch, args):
    model.eval()

    y_trues, y_preds = [], []
    total_loss, total_num, total_correct = 0.0, 0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, offsets, target in data_bar:
        data = data.to(args.device)
        offsets = offsets.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data, offsets)
        loss = criterion(output, target)

        total_loss += loss.item()
        total_num += target.size(0)
        pred = torch.argmax(output, dim=-1)
        y_trues.append(target)
        y_preds.append(pred)

        total_correct += torch.sum((pred == target).float()).item()
        accu = 100 * total_correct / total_num

        info = "Test  Epoch: [{}/{}] Loss: {:.4f} ".format(
            epoch, args.epochs, total_loss/len(data_loader))
        info += "Accu: {:.2f}".format(accu)
        data_bar.set_description(info)

    y_trues = torch.concatenate(y_trues).cpu().numpy()
    y_preds = torch.concatenate(y_preds).cpu().numpy()
    cm = confusion_matrix(y_trues, y_preds)
    
    return total_loss/len(data_loader), accu, cm


def main(args):

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.manual_seed(1243)

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
    ann_f = open(os.path.join(args.data_dir, 'annotations.txt'), newline='')
    reader = csv.reader(ann_f)
    seqdata = list(reader)
    sub_f = open(os.path.join(args.data_dir, 'subject_labels.txt'), newline='')
    reader = csv.reader(sub_f)
    labeldata = np.array(list(reader)).flatten()
    print('Data for %d subjects has been loaded' % len(labeldata))
    num_subjects = len(labeldata)

    args.vocab_size = 6     # 5 sleep stages + padding token (5)
    args.seg_seqlen = 60    # 30 eeg epochs
    
    args.writer = None
    with open(args.output_dir + "/args.json", 'w') as fid:
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        json.dump(args.__dict__, fid, indent=2, default=default)
    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log"))

    if args.start_fold <= 0:
        kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        splits_train, splits_test = [], []
        for (a, b) in kfold.split(np.arange(num_subjects)):
            splits_train.append(a)
            splits_test.append(b)
        np.savez(args.output_dir + '/splits.npz', splits_train=splits_train, splits_test=splits_test)
    else:
        splits = np.load(args.splits, allow_pickle=True)
        splits_train, splits_test = splits['splits_train'], splits['splits_test']

    # k-fold cross-validation
    train_accus, train_losses = np.zeros(args.folds), np.zeros(args.folds)
    test_accus,  test_losses  = np.zeros(args.folds), np.zeros(args.folds)
    test_cms, sensis, specis = np.zeros((args.folds, 2, 2)), np.zeros(args.folds), np.zeros(args.folds)
    for fold in range(args.start_fold, args.folds):

        idx_train, idx_test = splits_train[fold], splits_test[fold]

        train_seqdata = [seqdata[i] for i in idx_train]
        train_labeldata = [labeldata[i] for i in idx_train]
        test_seqdata = [seqdata[i] for i in idx_test]
        test_labeldata = [labeldata[i] for i in idx_test]

        train_dataset = SeqClassificationDataset(train_seqdata, train_labeldata)
        test_dataset = SeqClassificationDataset(test_seqdata, test_labeldata)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_batch,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_batch,
        )

        model = BaseNN(
            num_classes = 2,
            vocab_size = args.vocab_size,
            embed_dim = args.embed_dim,
        )
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        model = model.to(args.device)

        for epoch in range(args.epochs):
            start = time.time()
            utils.adjust_learning_rate(optimizer, epoch, args)
            lr = optimizer.param_groups[0]["lr"]

            train_losses[fold], train_accus[fold], _ = train_epoch(
                train_loader, model, criterion, optimizer, epoch, args)
            
            test_losses[fold], test_accus[fold], test_cms[fold] = evaluate(
                test_loader, model, criterion, epoch, args)
            
            sensis[fold] = test_cms[fold][0][0] / (test_cms[fold][0][0] + test_cms[fold][1][0]) # tp/(tp+fn)
            specis[fold] = test_cms[fold][1][1] / (test_cms[fold][1][1] + test_cms[fold][0][1]) # tn/(tn+fp)

            if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, epoch + 1,
                    is_best=False,
                    save_dir=os.path.join(args.output_dir, f"checkpoint/fold_{fold}"))

            if hasattr(args, 'writer') and args.writer:
                args.writer.add_scalar(f"Fold_{fold}/Accu/train", train_accus[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Accu/test",  test_accus[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Loss/train", train_losses[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Loss/test",  test_losses[fold], epoch)
                args.writer.add_scalar(f"Fold_{fold}/Misc/learning_rate", lr, epoch)

            print(f"Fold: {fold}, Epoch: {epoch}, "
                f"Train accu: {train_accus[fold]:.3f}, loss: {train_losses[fold]:.3f}, "
                f"Test accu: {test_accus[fold]:.3f}, loss: {test_losses[fold]:.3f}, "
                f"Sensitivity: {sensis[fold]:.3f}, Specificity: {specis[fold]:.3f}, "
                f"Epoch time = {time.time() - start:.3f}s")

    # Average over folds
    folds = [f"fold_{i}" for i in range(args.folds)] + ['average']
    train_accus = np.append(train_accus, np.mean(train_accus))
    train_losses = np.append(train_losses, np.mean(train_losses))
    test_accus  = np.append(test_accus, np.mean(test_accus))
    test_losses  = np.append(test_losses, np.mean(test_losses))
    sensis = np.append(sensis, np.mean(sensis))
    specis = np.append(specis, np.mean(specis))
    cm = np.sum(test_cms, axis=0)
    df_results = pd.DataFrame({
        'folds': folds,
        'train_accus': train_accus,
        'train_losses': train_losses,
        'test_accus' : test_accus,
        'test_losses' : test_losses,
        'sensitivity' : sensis,
        'specificity' : specis,
    })
    df_results.to_csv(os.path.join(args.output_dir, 'results_' + model._get_name() + '.csv'))
    with open(os.path.join(args.output_dir, 'confusion_matrix.txt'), 'w') as f:
        csv.writer(f, delimiter=' ').writerows(cm)
        f.close()
    print(df_results)
    print(cm)
            

if __name__ == "__main__":

    args = parser.parse_args()

    args.data_dir = '/home/yuty2009/data/eegdata/sleep/cap_sleepedf_seq/'
    args.output_dir = '/home/yuty2009/data/eegdata/sleep/cap_sleepedf_seq/output/'
    args.sm_dir = '/home/yuty2009/data/eegdata/sleep/shhs/'
    args.pretrained = None # = args.sm_dir + '/output/gpt/session_20230707233436_30_192_1_6/checkpoint/chkpt_0050.pth.tar'

    output_prefix = f"{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    main(args)
