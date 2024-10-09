
import mne
import yasa
import tqdm
import torch
import datetime
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import os,sys; sys.path.append(os.path.dirname(__file__)+"/../../")
import torchutils as utils
import datasets.sleepedfreader as sedfreader
import datasets.massreader as massreader
from models.gpt_transformers import GPTLM


sleep_datasets = {
    'sleepedf' : {
        'data_dir' : 'e:/eegdata/sleep/sleepedf153/sleep-cassette/',
        'output_dir' : 'e:/eegdata/sleep/sleepedf153/sleep-cassette/output/',
    },
    'mass' : {
        'data_dir' : '/home/yuty2009/data/eegdata/sleep/mass/',
        'output_dir' : '/home/yuty2009/data/eegdata/sleep/mass/output/',
    },
}

parser = argparse.ArgumentParser(description='Evaluate the Sleep Model')
parser.add_argument('-D', '--dataset', default='sleepedf', metavar='PATH',
                    help='dataset used')
args = parser.parse_args()

args.data_dir = sleep_datasets[args.dataset]['data_dir']
args.output_dir = sleep_datasets[args.dataset]['output_dir']

output_prefix = f"slm_yasa"
output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
if not hasattr(args, 'output_dir'):
    args.output_dir = args.data_dir
args.output_dir = os.path.join(args.output_dir, output_prefix)
os.makedirs(args.output_dir)
print("=> results will be saved to {}".format(args.output_dir))

args.sm_pretrained = 'output/gpt_shhs_pretrained/90_48_3_6.pth.tar'

args.seg_seqlen = 90
args.embed_dim = 48
args.num_layers = 3
args.num_heads = 6

# Data loading code
print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
if args.dataset == 'sleepedf':
    data, labels, subjects, ch_labels, keep_idxs, edf_paths = sedfreader.load_dataset_preprocessed(args.data_dir+'processed/', extra_info=True)
elif args.dataset == 'mass':
    data, labels, subjects, ch_labels, keep_idxs, edf_paths = massreader.load_dataset_preprocessed(args.data_dir+'processed/', extra_info=True)
else:
    raise NotImplementedError

print('Data for %d subjects has been loaded' % len(data))

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.vocab_size = 6 # 5 sleep stages + padding token (5)
sleep_model = GPTLM(
    vocab_size = args.vocab_size,
    max_seqlen = args.seg_seqlen,
    embed_dim = args.embed_dim,
    num_layers = args.num_layers, 
    num_heads = args.num_heads,
)
utils.load_checkpoint(args.sm_pretrained, sleep_model, strict=True)
sleep_model = sleep_model.to(args.device)
sleep_model.eval()

alpha = 0.1
ngram = 30
ytrues, ypreds, ycorrecteds = [], [], []
num_total, num_correct = 0, 0
for i in tqdm.tqdm(range(len(subjects))):

    signal = mne.io.read_raw_edf(str(edf_paths[i]), preload=True, verbose=False)
    print('The channels are:', signal.ch_names)
    print('The sampling frequency is:', signal.info['sfreq'])

    sample_f = 100
    channels = signal.info['ch_names']
    # signal = signal.set_eeg_reference(ref_channels=["A1"])
    # signal_notched = signal.notch_filter(freqs=50, notch_widths=2)
    # signal_processed = signal_notched.filter(l_freq=0.3, h_freq=35)
    signal_processed = signal.resample(sfreq=sample_f)
    signal_data = signal_processed.get_data()

    eog_name, emg_name = None, None
    for ch_name in channels:
        if ch_name.startswith('EOG'):
            eog_name = ch_name
            break
    for ch_name in channels:
        if ch_name.startswith('EMG'):
            emg_name = ch_name
            break

    channel_stage = str(ch_labels[i]).split(',')[0]
    sls = yasa.SleepStaging(signal_processed, eeg_name=channel_stage, eog_name=eog_name, emg_name=emg_name)
    # data_sub = data[i]
    # sls = SleepStaging(data_sub, eeg_name='EEG', eog_name=None, emg_name=None)
    yprob = sls.predict_proba()
    cols = ['W', 'N1', 'N2', 'N3', 'R'] # ['N1', 'N2', 'N3', 'R', 'W'] => ['W', 'N1', 'N2', 'N3', 'R']
    yprob = yprob[cols].to_numpy()
    ypred = np.argmax(yprob, axis=-1)
    ytrue = labels[i]
    ytrues += ytrue.tolist()

    keep_idx = keep_idxs[i]
    yprob = yprob[keep_idx]
    ypred = ypred[keep_idx]
    ypreds += ypred.tolist()
    assert len(ypred) == len(ytrue)

    yprob = torch.FloatTensor(yprob).to(args.device)
    ycorrected = sleep_model.correct(yprob, ngram=ngram, lm_weight=alpha).flatten()
    ycorrecteds += ycorrected.tolist()

accu_raw = accuracy_score(ytrues, ypreds)
mf1_raw = f1_score(ytrues, ypreds, average='macro')
kappa_raw = cohen_kappa_score(ytrues, ypreds)
accu_sm = accuracy_score(ytrues, ycorrecteds)
mf1_sm = f1_score(ytrues, ycorrecteds, average='macro')
kappa_sm = cohen_kappa_score(ytrues, ycorrecteds)

savepath = os.path.join(args.output_dir, 'result.txt')
out_f = open(savepath, 'w')

out_str = f"accu_raw: {accu_raw}, accu_sm: {accu_sm}, " \
        f"mf1_raw: {mf1_raw}, mf1_sm: {mf1_sm}, " \
        f"kappa_raw: {kappa_raw}, kappa_sm: {kappa_sm}"
out_f.write(out_str + "\n")
print(out_str)
