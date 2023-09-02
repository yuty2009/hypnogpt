
import os
import csv
import time
import tqdm
import datetime
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from timm.optim.optim_factory import param_groups_weight_decay

import torchutils as utils
from gpt_transformers import GPTLM


sleep_datasets = {
    'shhs' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/shhs/processed/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/shhs/output/',
    }
}

parser = argparse.ArgumentParser(description='Train and evaluate a GPT model')
parser.add_argument('-D', '--dataset', default='shhs', metavar='PATH',
                help='dataset used')
parser.add_argument('--pretrained', 
                    default='',
                    metavar='PATH', help='path to pretrained model (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='gpt',
                help='model architecture (default: gpt)')
parser.add_argument('--max_seqlen', default=120, type=int, metavar='N',
                    help='maximum acceptable sequence length (default: 30)')
parser.add_argument('--embed_dim', default=192, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=2, type=int, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('--num_heads', default=6, type=int, metavar='N',
                    help='number of heads for multi-head attention (default: 6)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
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

# create a mapping from characters to integers
stoi = { "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4 }
itos = { 0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM" }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Auto-regressive sequence dataset
class ARSeqDataset(torch.utils.data.Dataset):
    # sequence: the whole tokenized sequence, e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # bsz: block size, e.g. 5
    # then using the first 5 tokens as input, and the 2-6 tokens as label
    def __init__(self, sequence, bsz):
        self.bsz = bsz
        self.sequence = [int(s) for s in sequence]
        self.len = len(sequence) - bsz

    def __getitem__(self, index):
        data = torch.LongTensor(self.sequence[index : index + self.bsz])
        label = torch.LongTensor(self.sequence[index + 1 : index + self.bsz + 1])
        return data, label

    def __len__(self):
        return self.len
    

def train_epoch(data_loader, model, optimizer, epoch, args):
    model.train()
    total_loss, total_num = 0.0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        loss = model(input_ids=data, labels=target)[0]
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update metrics
        total_loss += loss.item()
        total_num += data.size(0)
        # show progress
        info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
            epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
        data_bar.set_description(info)
    return total_loss/len(data_loader)


def evaluate(data_loader, model, epoch, args):
    model.eval()
    total_loss, total_num = 0.0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        loss = model(input_ids=data, labels=target)[0]
        # update metrics
        total_loss += loss.item()
        total_num += data.size(0)
        # show progress
        info = "Test  Epoch: [{}/{}] Loss: {:.4f} ".format(
            epoch, args.epochs, total_loss/len(data_loader))
        data_bar.set_description(info)
    return total_loss/len(data_loader)


def main(args):

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1243)

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
    ann_f = open(os.path.join(args.data_dir, 'annotations.txt'), newline='')
    reader = csv.reader(ann_f)
    labeldata = list(reader)
    print('Data for %d subjects has been loaded' % len(labeldata))
    num_subjects = len(labeldata)
    import numpy as np
    tokens = np.unique(labeldata[0])
    for ldata in labeldata:
        if len(np.unique(ldata)) > len(tokens):
            raise NotImplementedError

    args.vocab_size = 6     # 5 sleep stages + padding token (5)
    args.block_size = args.max_seqlen    # 30 eeg epochs
    
    idx_all = list(range(0, num_subjects))
    idx_train, idx_test = idx_all, idx_all[:39]
    trainsets = [ARSeqDataset(labeldata[i], args.block_size) for i in idx_train]
    testsets  = [ARSeqDataset(labeldata[i], args.block_size) for i in idx_test]

    train_dataset = torch.utils.data.ConcatDataset(trainsets)
    test_dataset = torch.utils.data.ConcatDataset(testsets)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    model = GPTLM(
        vocab_size=args.vocab_size,
        max_seqlen=args.max_seqlen,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = model.to(args.device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")
    param_groups = param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if hasattr(args, 'pretrained') and args.pretrained:
        utils.load_checkpoint(args.pretrained, model, optimizer, args)

    if not args.evaluate:
    
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log"))

        for epoch in range(args.epochs):
            start = time.time()
            utils.adjust_learning_rate(optimizer, epoch, args)
            lr = optimizer.param_groups[0]["lr"]

            train_loss = train_epoch(train_loader, model, optimizer, epoch, args)
            test_loss = evaluate(test_loader, model, epoch, args)

            if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, epoch + 1,
                    is_best=False,
                    save_dir=os.path.join(args.output_dir, f"checkpoint"))

            if hasattr(args, 'writer') and args.writer:
                args.writer.add_scalar(f"Loss/train", train_loss, epoch)
                args.writer.add_scalar(f"Loss/test",  test_loss, epoch)
                args.writer.add_scalar(f"Misc/learning_rate", lr, epoch)

            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}, "
                f"Epoch time = {time.time() - start:.3f}s")

    context = torch.tensor([[0]], dtype=torch.long, device=args.device)
    next_probs = model.next_log_probs(context)
    print(decode(context[0].cpu().numpy()))
    print(next_probs)

if __name__ == "__main__":

    args = parser.parse_args()

    args.data_dir = sleep_datasets[args.dataset]['data_dir']
    args.output_dir = sleep_datasets[args.dataset]['output_dir']

    output_prefix = f"{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_prefix += f"_{args.max_seqlen}_{args.embed_dim}_{args.num_layers}_{args.num_heads}"
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    main(args)
