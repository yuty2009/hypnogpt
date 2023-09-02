
import torch
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

import torchutils as utils
from eegreader import TransformEpoch, EEGDataset, SeqEEGDataset
from datasets import sleepedfreader as sedfreader
from datasets import massreader
from datasets import shhsreader
# from models.sleepnet import TinySleepNet
from models.seqsleepnet import TinySleepNet
from gpt_transformers import GPTLM


sleep_datasets = {
    'sleepedf' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/sleepedf153/sleep-cassette/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/sleepedf153/sleep-cassette/output/',
    },
    'mass' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/mass/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/mass/output/',
    },
    'shhs' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/shhs/processed/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/shhs/output/',
    }
}

parser = argparse.ArgumentParser(description='Evaluate the Sleep Model')
args = parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.dataset = 'mass'
args.arch = 'tinysleepnet'
args.data_dir = sleep_datasets[args.dataset]['data_dir']
args.output_dir = sleep_datasets[args.dataset]['output_dir']
# args.slm_dir = '/home/yuty2009/data/eegdata/sleep/sleepedf153/sleep-cassette/'
args.slm_dir = '/home/yuty2009/data/eegdata/sleep/shhs/'
# args.pretrained_dir = args.data_dir + '/output/tinysleepnet/session_20230520015213/checkpoint/' # sleepedf
# args.pretrained_dir = args.data_dir + '/output/seq_tinysleepnet/session_20230503123518/checkpoint/' # sleepedf
# args.pretrained_dir = args.data_dir + '/output/tinysleepnet/session_20230705213750/checkpoint/' # mass
args.pretrained_dir = args.data_dir + '/output/seq_tinysleepnet/session_20230602212437/checkpoint/' # mass
# args.splits = args.data_dir + '/output/tinysleepnet/session_20230520015213/splits.npz' # sleepedf
# args.splits = args.data_dir + '/output/seq_tinysleepnet/session_20230503123518/splits.npz' # sleepedf
# args.splits = args.data_dir + '/output/tinysleepnet/session_20230705213750/splits.npz' # mass
args.splits = args.data_dir + '/output/seq_tinysleepnet/session_20230602212437/splits.npz' # mass
args.slm_pretrained = args.slm_dir + '/output/gpt/session_20230722001546_60_192_2_6/checkpoint/chkpt_0050.pth.tar'
# /home/yuty2009/data/eegdata/sleep/shhs/output/gpt/session_20230629004115_30_192_1_6
# session_20230603144209_30_192_1_6: minigpt, max_seqlen = 30, embed_dim = 192, num_layers = 1, num_heads = 6,
# session_20230603004639_30_384_1_6: minigpt, max_seqlen = 30, embed_dim = 384, num_layers = 1, num_heads = 6,
# session_20230603000504_1500_384_1_6: minigpt, max_seqlen = 1500, embed_dim = 384, num_layers = 1, num_heads = 6,
# session_20230601233153_30_384_1_6: gpt_transformers, max_seqlen = 30, embed_dim = 384, num_layers = 1, num_heads = 6,
# session_20230602105043_1500_384_1_6: gpt_transformers, max_seqlen = 1500, embed_dim = 384, num_layers = 1, num_heads = 6,
# /home/yuty2009/data/eegdata/sleep/shhs/output/gpt/session_20230703165808_30_192_1_6
# session_20230703165808_30_192_1_6: gpt_transformers, max_seqlen = 30, embed_dim = 192, num_layers = 1, num_heads = 6,
# session_20230705235154_30_192_1_6: gpt_transformers, max_seqlen = 30, embed_dim = 192, num_layers = 1, num_heads = 6, full
# session_20230724195320_15_192_1_6: gpt_transformers, max_seqlen = 15, embed_dim = 192, num_layers = 1, num_heads = 6, vocab6
# session_20230707233436_30_192_1_6: gpt_transformers, max_seqlen = 30, embed_dim = 192, num_layers = 1, num_heads = 6, vocab6
# session_20230709015037_60_192_1_6: gpt_transformers, max_seqlen = 60, embed_dim = 192, num_layers = 1, num_heads = 6, vocab6
# session_20230710141711_90_192_1_6: gpt_transformers, max_seqlen = 90, embed_dim = 192, num_layers = 1, num_heads = 6, vocab6
# session_20230721101912_120_192_1_6: gpt_transformers, max_seqlen = 120, embed_dim = 192, num_layers = 1, num_heads = 6, vocab6
# session_20230721102032_180_192_1_6: gpt_transformers, max_seqlen = 180, embed_dim = 192, num_layers = 1, num_heads = 6, vocab6
# session_20230724195357_15_192_2_6: gpt_transformers, max_seqlen = 15, embed_dim = 192, num_layers = 2, num_heads = 6, vocab6
# session_20230721114701_30_192_2_6: gpt_transformers, max_seqlen = 30, embed_dim = 192, num_layers = 2, num_heads = 6, vocab6
# session_20230722001546_60_192_2_6: gpt_transformers, max_seqlen = 60, embed_dim = 192, num_layers = 2, num_heads = 6, vocab6
# session_20230722001546_90_192_2_6: gpt_transformers, max_seqlen = 90, embed_dim = 192, num_layers = 2, num_heads = 6, vocab6
# session_20230722030517_120_192_2_6: gpt_transformers, max_seqlen = 120, embed_dim = 192, num_layers = 2, num_heads = 6, vocab6
# session_20230722064619_180_192_2_6: gpt_transformers, max_seqlen = 180, embed_dim = 192, num_layers = 2, num_heads = 6, vocab6
# session_20230725100916_15_192_6_6: gpt_transformers, max_seqlen = 15, embed_dim = 192, num_layers = 6, num_heads = 6, vocab6
# session_20230722193353_30_192_6_6: gpt_transformers, max_seqlen = 30, embed_dim = 192, num_layers = 6, num_heads = 6, vocab6
# session_20230722193353_60_192_6_6: gpt_transformers, max_seqlen = 60, embed_dim = 192, num_layers = 6, num_heads = 6, vocab6
# session_20230722193359_90_192_6_6: gpt_transformers, max_seqlen = 90, embed_dim = 192, num_layers = 6, num_heads = 6, vocab6
# session_20230722220709_120_192_6_6: gpt_transformers, max_seqlen = 120, embed_dim = 192, num_layers = 6, num_heads = 6, vocab6
# session_20230723130318_180_192_6_6: gpt_transformers, max_seqlen = 180, embed_dim = 192, num_layers = 6, num_heads = 6, vocab6

# Data loading code
print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
if args.dataset == 'sleepedf':
    data, labels, subjects = sedfreader.load_dataset_preprocessed(args.data_dir+'processed/')
elif args.dataset == 'mass':
    data, labels, subjects = massreader.load_dataset_preprocessed(args.data_dir+'processed/')
elif args.dataset == 'shhs':
    data, labels, subjects = shhsreader.load_dataset_preprocessed(args.data_dir+'processed/')
else:
    raise NotImplementedError

print('Data for %d subjects has been loaded' % len(data))
num_subjects = len(data)
args.num_classes = 5
args.n_wavlen = data[0].shape[-2]
args.seq_len = 15
args.topk = (1, 5)
args.epochs = 0

tf_epoch = TransformEpoch()

# all_datasets = [EEGDataset(data[i], labels[i], tf_epoch) for i in range(num_subjects)]
all_datasets = [SeqEEGDataset(data[i], labels[i], args.seq_len, tf_epoch) for i in range(num_subjects)]

# create model
args.patch_size = 20
args.embed_dim = 192
print("=> creating sleep model")
if args.arch in ['tinysleepnet', 'TinySleepNet']:
    model = TinySleepNet(
        n_classes = args.num_classes,
        n_seqlen = args.seq_len,
        n_timepoints = args.n_wavlen,
    )

args.vocab_size = 6 # 5 sleep stages + padding token (5)
sleep_model = GPTLM(
    vocab_size = args.vocab_size,
    max_seqlen = 60,
    embed_dim = 192,
    num_layers = 2, 
    num_heads = 6
)
utils.load_checkpoint(args.slm_pretrained, sleep_model)
sleep_model = sleep_model.to(args.device)
sleep_model.eval()

# k-fold cross-validation
args.folds = 10
args.start_fold = 0
splits = np.load(args.splits, allow_pickle=True)
splits_train, splits_test = splits['splits_train'], splits['splits_test']

# alpha=0.3, ngram=10, accu_sm = 0.82509, 1 layers, 6 heads
alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
ngrams = [5, 10, 15, 20, 25, 30]
accu_raw, accu_sm = np.zeros(args.folds), np.zeros((args.folds, len(alphas), len(ngrams)))
mf1_raw, mf1_sm = np.zeros(args.folds), np.zeros((args.folds, len(alphas), len(ngrams)))
kappa_raw, kappa_sm = np.zeros(args.folds), np.zeros((args.folds, len(alphas), len(ngrams)))
for fold in range(args.start_fold, args.folds):

    idx_train, idx_test = splits_train[fold], splits_test[fold]
    
    utils.load_checkpoint(args.pretrained_dir + f"fold_{fold}/chkpt_0200.pth.tar", model)
    model = model.to(args.device)
    model.eval()

    test_target, test_pred = [], []
    test_corrected = [[[] for j in range(len(ngrams))] for i in range(len(alphas))]
    for sub in idx_test:
        test_loader = torch.utils.data.DataLoader(all_datasets[sub], batch_size=256)
        logits_pred_sub = []
        labels_true_sub = []
        for data, target in test_loader:
            data = data.to(args.device)
            target = target.to(args.device)
            # compute output
            output = model(data)
            logits_pred_sub.append(output)
            labels_true_sub.append(target)
        logits_pred_sub = torch.concatenate(logits_pred_sub)
        labels_true_sub = torch.concatenate(labels_true_sub)
        labels_pred_sub = torch.argmax(logits_pred_sub, dim=-1)

        test_target += list(labels_true_sub.cpu().numpy())
        test_pred += list(labels_pred_sub.cpu().numpy())

        for i in range(len(alphas)):
            for j in range(len(ngrams)):
                labels_corrected_sub = sleep_model.correct(logits_pred_sub, ngram=ngrams[j], lm_weight=alphas[i]).flatten()
                test_corrected[i][j] += list(labels_corrected_sub.cpu().numpy())

    accu_raw[fold] = accuracy_score(test_target, test_pred)
    mf1_raw[fold] = f1_score(test_target, test_pred, average='macro')
    kappa_raw[fold] = cohen_kappa_score(test_target, test_pred)
    for i in range(len(alphas)):
        for j in range(len(ngrams)):
            accu_sm[fold][i][j] = accuracy_score(test_target, test_corrected[i][j])
            mf1_sm[fold][i][j] = f1_score(test_target, test_corrected[i][j], average='macro')
            kappa_sm[fold][i][j] = cohen_kappa_score(test_target, test_corrected[i][j])

    print(f"Fold {fold} "
          f"accu_raw: {accu_raw[fold]}, accu_sm: {accu_sm[fold][0][0]}, "
          f"mf1_raw: {mf1_raw[fold]}, mf1_sm: {mf1_sm[fold][0][0]}, "
          f"kappa_raw: {kappa_raw[fold]}, kappa_sm: {kappa_sm[fold][0][0]}")

print(
    f"{args.folds}-fold Cross-Validation "
    f"alpha: 0, ngram: 0, accu_raw: {np.mean(accu_raw)}, mf1_raw: {np.mean(mf1_raw)}, kappa_raw: {np.mean(kappa_raw)}"
)
for i in range(len(alphas)):
    for j in range(len(ngrams)):
        print(
            f"alpha: {alphas[i]}, ngram: {ngrams[j]}, "
            f"accu_sm: {np.mean(accu_sm, 0)[i][j]}, mf1_sm: {np.mean(mf1_sm, 0)[i][j]}, kappa_sm: {np.mean(kappa_sm, 0)[i][j]}"
        )
