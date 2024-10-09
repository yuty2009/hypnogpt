
import tqdm
import torch
import numpy as np


class SeqClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = []
        for seq_raw in sequences:
            seq = [int(s) for s in seq_raw]
            self.sequences.append(seq)
        self.labels = labels
        self.len = len(labels)

    def __getitem__(self, index):
        sequence = torch.LongTensor(self.sequences[index])
        label = self.labels[index]
        return sequence, label

    def __len__(self):
        return self.len
    

class SleepSequenceCollator(object):

    def __init__(self, seg_seqlen=30, pad_token_id=5, padding_side='right'):
        self.seg_seqlen = seg_seqlen
        self.pad_token_id = pad_token_id
        self.padding_side = padding_side

    def __call__(self, batch):
        batch_x = []
        batch_y = []
        for (x, y) in batch:
            batch_x.append(x)
            batch_y.append(y)

        batch_size = len(batch_x)
        lengths = [x.size(-1) for x in batch_x]
        max_length = int(np.ceil(max(lengths) / self.seg_seqlen) * self.seg_seqlen)
        
        sequences = self.pad_token_id * torch.ones([batch_size, max_length], dtype=batch_x[0].dtype)
        masks = torch.zeros(batch_size, max_length)
        labels = torch.LongTensor(batch_y)

        for k in range(batch_size):
            if self.padding_side == 'right':
                sequences[k, :lengths[k]] = batch_x[k]
                masks[k, :lengths[k]] = 1
            elif self.padding_side == 'left':
                sequences[k, -lengths[k]:] = batch_x[k]
                masks[k, -lengths[k]:] = 1
            else:
                raise ValueError("Padding side should be either left or right")

        sequences = sequences.reshape(batch_size, -1, self.seg_seqlen)
        masks = masks.reshape(batch_size, -1, self.seg_seqlen)
        return sequences, masks, labels


def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    
    y_trues, y_preds, y_probs = [], [], []
    total_loss, total_num, total_correct = 0.0, 0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, masks, target in data_bar:
        data = data.to(args.device)
        masks = masks.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data, masks)[0]
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_num += target.size(0)
        pred = torch.argmax(output, dim=-1)
        prob = torch.nn.functional.softmax(output, dim=-1)
        y_trues.append(target)
        y_preds.append(pred)
        y_probs.append(prob)

        total_correct += torch.sum((pred == target).float()).item()
        accu = 100 * total_correct / total_num

        info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
            epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
        info += "Accu: {:.2f}".format(accu) 
        data_bar.set_description(info)
    
    y_trues = torch.concatenate(y_trues).cpu().numpy()
    y_preds = torch.concatenate(y_preds).cpu().numpy()
    y_probs = torch.cat(y_probs, dim=0).detach().cpu().numpy()
    
    return total_loss/len(data_loader), accu, y_trues, y_probs


def evaluate(data_loader, model, criterion, epoch, args):
    model.eval()

    y_trues, y_preds, y_probs = [], [], []
    total_loss, total_num, total_correct = 0.0, 0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, masks, target in data_bar:
        data = data.to(args.device)
        masks = masks.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data, masks)[0]
        loss = criterion(output, target)

        total_loss += loss.item()
        total_num += target.size(0)
        pred = torch.argmax(output, dim=-1)
        prob = torch.nn.functional.softmax(output, dim=-1)
        y_trues.append(target)
        y_preds.append(pred)
        y_probs.append(prob)

        total_correct += torch.sum((pred == target).float()).item()
        accu = 100 * total_correct / total_num

        info = "Test  Epoch: [{}/{}] Loss: {:.4f} ".format(
            epoch, args.epochs, total_loss/len(data_loader))
        info += "Accu: {:.2f}".format(accu)
        data_bar.set_description(info)

    y_trues = torch.concatenate(y_trues).cpu().numpy()
    y_preds = torch.concatenate(y_preds).cpu().numpy()
    y_probs = torch.cat(y_probs, dim=0).detach().cpu().numpy()
    
    return total_loss/len(data_loader), accu, y_trues, y_probs


def sensitivity_specificity(conf_matrix):
    true_positives = conf_matrix[1, 1]
    false_negatives = conf_matrix[1, 0]
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]

    if true_positives + false_negatives == 0:
        true_positives += 1
    if true_negatives + false_positives == 0:
        true_negatives += 1

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    return sensitivity, specificity


def crop_sleep_period(seqdata, w_edge_mins=30, wake_id='0'):
    """ Select only sleep periods"""
    seqdata_new = []
    for y in seqdata:
        y = np.asarray(y)
        nw_idx = np.where(y != wake_id)[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx+1)
        y = y[select_idx]
        seqdata_new.append(y)
    return seqdata_new
