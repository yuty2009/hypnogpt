
import os
import glob
import pyedflib
import numpy as np


_available_dataset = [
    'sleep-edf-v1',
    'sleep-edf-ex',
]

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}

# Label values
W = 0       # Stage AWAKE
N1 = 1      # Stage N1
N2 = 2      # Stage N2
N3 = 3      # Stage N3
REM = 4     # Stage REM
MOVE = 5    # Movement
UNK = 6     # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK,
}


def load_eegdata(setname, datapath, subject):
    assert setname in _available_dataset, 'Unknown dataset name ' + setname
    if setname == 'sleepedf':
        filepath = os.path.join(datapath, subject+'.rec')
        labelpath = os.path.join(datapath, subject+'.hyp')
        data, target = load_eegdata_sleepedfx(filepath, labelpath)
    if setname == 'sleepedfx':
        filepath = os.path.join(datapath, subject+'-PSG.edf')
        labelpath = os.path.join(datapath, subject+'-Hypnogram.edf')
        data, target = load_eegdata_sleepedfx(filepath, labelpath)
    return data, target


def load_eegdata_sleepedf(rec_fname, hyp_fname):
    data = []
    target = []
    return data, target


def load_eegdata_sleepedfx(psg_fname, ann_fname, select_ch='EEG Fpz-Cz'):
    """
    https://github.com/akaraspt/tinysleepnet
    """
    
    psg_f = pyedflib.EdfReader(psg_fname)
    ann_f = pyedflib.EdfReader(ann_fname)

    assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
    start_datetime = psg_f.getStartdatetime()

    file_duration = psg_f.getFileDuration()
    epoch_duration = psg_f.datarecord_duration
    if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
        epoch_duration = epoch_duration / 2

    # Extract signal from the selected channel
    ch_names = psg_f.getSignalLabels()
    ch_samples = psg_f.getNSamples()
    select_ch_idx = -1
    for s in range(psg_f.signals_in_file):
        if ch_names[s] == select_ch:
            select_ch_idx = s
            break
    if select_ch_idx == -1:
        raise Exception("Channel not found.")
    sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
    n_epoch_samples = int(epoch_duration * sampling_rate)
    signals = psg_f.readSignal(select_ch_idx).reshape(-1, n_epoch_samples)

    # Sanity check
    n_epochs = psg_f.datarecords_in_file
    if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
        n_epochs = n_epochs * 2
    assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

    # Generate labels from onset and duration annotation
    labels = []
    total_duration = 0
    ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
    for a in range(len(ann_stages)):
        onset_sec = int(ann_onsets[a])
        duration_sec = int(ann_durations[a])
        ann_str = "".join(ann_stages[a])

        # Sanity check
        assert onset_sec == total_duration

        # Get label value
        label = ann2label[ann_str]

        # Compute # of epoch for this stage
        if duration_sec % epoch_duration != 0:
            raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
        duration_epoch = int(duration_sec / epoch_duration)

        # Generate sleep stage labels
        label_epoch = np.ones(duration_epoch, dtype=int) * label
        labels.append(label_epoch)

        total_duration += duration_sec

    labels = np.hstack(labels)

    # Remove annotations that are longer than the recorded signals
    labels = labels[:len(signals)]

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)
    y_full = y.copy()
    keep_idx = np.arange(len(y_full))

    # Select only sleep periods
    w_edge_mins = 30
    nw_idx = np.where(y != stage_dict["W"])[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    if start_idx < 0: start_idx = 0
    if end_idx >= len(y): end_idx = len(y) - 1
    select_idx = np.arange(start_idx, end_idx+1)
    x = x[select_idx]
    y = y[select_idx]
    keep_idx = keep_idx[select_idx]

    # Remove movement and unknown
    remove_idx = np.where(y >= 5)[0]
    if len(remove_idx) > 0:
        select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        x = x[select_idx]
        y = y[select_idx]
        keep_idx = keep_idx[select_idx]

    remove_idx = np.where(y_full >= 5)[0]
    if len(remove_idx) > 0:
        select_idx = np.setdiff1d(np.arange(len(y_full)), remove_idx)
        y_full = y_full[select_idx]

    # Save
    data_dict = {
        "x": x, 
        "y": y, 
        "y_full": y_full,
        "keep_idx": keep_idx,
        "fs": sampling_rate,
        "ch_label": select_ch,
        "start_datetime": start_datetime,
        "file_duration": file_duration,
        "epoch_duration": epoch_duration,
        "n_all_epochs": n_epochs,
        "n_epochs": len(x),
        "edf_path": psg_fname,
    }

    return data_dict


def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
        keep_idx = f['keep_idx']
        edf_path = f['edf_path']
    return data, labels, sampling_rate, keep_idx, edf_path

def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    keep_idxs = []
    edf_paths = []
    fs = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate, keep_idx, edf_path = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # # Reshape the data to match the input of the model - conv2d
        tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data.append(tmp_data)
        labels.append(tmp_labels)
        keep_idxs.append(keep_idx)
        edf_paths.append(edf_path)

    return data, labels, keep_idxs, edf_paths

def load_subdata_preprocessed(datapath, subject):
    npz_f = os.path.join(datapath, subject+'.npz')
    data, labels, fs, _, _ = load_npz_file(npz_f)
    return data, labels

def load_dataset_preprocessed(datapath, subsets=None, n_subjects=None, extra_info=False):
    if isinstance(subsets, str):
        subsets = [subsets]
    npzfiles = glob.glob(os.path.join(datapath, "*.npz"))
    npzfiles.sort()
    if n_subjects is not None:
        npzfiles = npzfiles[:n_subjects]
    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]
    data, labels, keep_idxs, edf_paths = load_npz_list_files(npzfiles)
    if extra_info:
        return data, labels, subjects, keep_idxs, edf_paths
    else:
        return data, labels, subjects
    

def load_npzlist_preprocessed(datapath, subsets=None, n_subjects=None):
    if isinstance(subsets, str):
        subsets = [subsets]
    npzfiles = glob.glob(os.path.join(datapath, "*.npz"))
    npzfiles.sort()
    if n_subjects is not None:
        npzfiles = npzfiles[:n_subjects]
    return npzfiles


if __name__ == '__main__':

    datapath = '/home/yuty2009/data/eegdata/sleep/sleepedf153/sleep-cassette/'
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    psg_fnames = glob.glob(os.path.join(datapath, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(datapath, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()

    annotations = []
    ann_f = open(datapath+'annotations.txt', 'w')
    ann_f_s1 = open(datapath+'annotations_s1.txt', 'w')
    for i in range(len(psg_fnames)):

        subject = os.path.basename(psg_fnames[i])[:-8]
        session = int(subject[-3])

        print('Load and extract continuous EEG into epochs for subject '+subject)
        data_dict = load_eegdata_sleepedfx(psg_fnames[i], ann_fnames[i])
        annotations.append(data_dict["y_full"])
        ann_f.write(",".join([f"{ann}" for ann in data_dict["y_full"]]) + "\n")
        if session == 1:
            ann_f_s1.write(",".join([f"{ann}" for ann in data_dict["y_full"]]) + "\n")

        np.savez(datapath+'processed/'+subject+'.npz', **data_dict)
    ann_f.close()
    ann_f_s1.close()
