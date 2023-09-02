
import os
import math
import tqdm
import torch
import numpy as np
        

def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    total_loss, total_num = 0.0, 0
    if not hasattr(args, 'topk'): args.topk = (1,)
    total_corrects = torch.zeros(len(args.topk), dtype=torch.float)

    data_bar = tqdm.tqdm(data_loader)

    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data)
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_num += data.size(0)
        preds = torch.argsort(output, dim=-1, descending=True)
        for i, k in enumerate(args.topk):
                total_corrects[i] += torch.sum((preds[:, 0:k] \
                    == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        accuks = 100 * total_corrects / total_num

        info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
            epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
        info += ' '.join(["Acc@{}: {:.2f}".format(k, accuk) 
                        for k, accuk in zip(args.topk, accuks)])
        data_bar.set_description(info)
    
    return [total_loss/len(data_loader)] + [accuk for accuk in accuks]


def evaluate(data_loader, model, criterion, epoch, args):
    model.eval()
    total_loss, total_num = 0.0, 0
    if not hasattr(args, 'topk'): args.topk = (1,)
    total_corrects = torch.zeros(len(args.topk), dtype=torch.float)

    data_bar = tqdm.tqdm(data_loader)

    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        total_num += data.size(0)
        preds = torch.argsort(output, dim=-1, descending=True)
        for i, k in enumerate(args.topk):
                total_corrects[i] += torch.sum((preds[:, 0:k] \
                    == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        accuks = 100 * total_corrects / total_num

        info = "Test  Epoch: [{}/{}] Loss: {:.4f} ".format(
            epoch, args.epochs, total_loss/len(data_loader))
        info += ' '.join(["Acc@{}: {:.2f}".format(k, accuk) 
                        for k, accuk in zip(args.topk, accuks)])
        data_bar.set_description(info)
    
    return [total_loss/len(data_loader)] + [accuk for accuk in accuks]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule after warmup"""
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = 0
    if not hasattr(args, 'min_lr'):
        args.min_lr = 0.
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr
        if args.schedule in ['cos', 'cosine']:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) # without warmup
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.schedule in ['step', 'stepwise']:  # stepwise lr schedule
            for milestone in args.lr_drop:
                lr *= 0.1 if epoch >= int(milestone * args.epochs) else 1.
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def change_learning_rate(optimizer, lr):
    """ Set the learning rate to a fixed value """
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def load_checkpoint(ckptpath, model, optimizer=None, args=None, strict=False):
    if os.path.isfile(ckptpath):
        checkpoint = torch.load(ckptpath, map_location='cpu')
        state_dict = convert_state_dict(checkpoint['state_dict'])
        msg = model.load_state_dict(state_dict, strict=strict)
        print(msg.missing_keys)
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(args.device)
        if args is not None:
            args.start_epoch = 0
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(ckptpath, args.start_epoch))
        else:
            print("=> loaded checkpoint '{}'".format(ckptpath))
    else:
        print("=> no checkpoint found at '{}'".format(ckptpath))


def save_checkpoint(state, epoch, is_best, save_dir='./'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(
        save_dir, 'chkpt_{:04d}.pth.tar'.format(epoch)
        )
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best.pth.tar')
        torch.save(state, best_path)


def convert_state_dict(state_dict):
    firstkey = next(iter(state_dict))
    if firstkey.startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.endswith('total_ops') and not k.endswith('total_params'):
                name = k[7:] # 7 = len('module.')
                new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
