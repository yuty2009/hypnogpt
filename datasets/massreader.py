
import os
import glob
import pyedflib
import numpy as np
from scipy import interpolate


# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

# Label values
W = 0       # Stage AWAKE
N1 = 1      # Stage N1
N2 = 2      # Stage N2
N3 = 3      # Stage N3
REM = 4     # Stage REM
MOVE = 5    # Movement
UNKNOWN = 5     # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNKNOWN": UNKNOWN,
}


def resample(signal, signal_frequency, target_frequency):
    resampling_ratio = signal_frequency / target_frequency
    x_base = np.arange(0, len(signal))

    interpolator = interpolate.interp1d(x_base, signal, axis=0, bounds_error=False, fill_value='extrapolate',)

    x_interp = np.arange(0, len(signal), resampling_ratio)

    signal_duration = signal.shape[0] / signal_frequency
    resampled_length = round(signal_duration * target_frequency)
    resampled_signal = interpolator(x_interp)
    if len(resampled_signal) < resampled_length:
        padding = np.zeros((resampled_length - len(resampled_signal), signal.shape[-1]))
        resampled_signal = np.concatenate([resampled_signal, padding])

    return resampled_signal


def load_eegdata_mass(psg_fname, ann_fname, select_ch=[['EEG C3', 'EEG A2']], target_fs=100.):

    psg_f = pyedflib.EdfReader(psg_fname)
    ann_f = pyedflib.EdfReader(ann_fname)

    # assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
    start_datetime = psg_f.getStartdatetime()

    file_duration = psg_f.getFileDuration()
    epoch_duration = 30.0 # psg_f.datarecord_duration
    
    # Generate labels from onset and duration annotation
    labels = []
    ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()
    total_duration = ann_onsets[0]
    for a in range(len(ann_stages)):
        onset_sec = int(round(ann_onsets[a]))
        duration_sec = int(round(ann_durations[a]))
        ann_str = "".join(ann_stages[a])

        # Sanity check
        assert onset_sec == int(round(total_duration))

        # Get label value
        label = ann2label[ann_str]

        # Compute # of epoch for this stage
        epoch_duration_actual = epoch_duration
        overlap_left, overlap_right = 0.0, 0.0
        if duration_sec == 20.0: # MASS2 duration is 20 sec
            epoch_duration_actual = 20.0
            overlap_left, overlap_right = 5.0, 5.0
        duration_epoch = int(round(duration_sec / epoch_duration_actual))

        # Generate sleep stage labels
        label_epoch = np.ones(duration_epoch, dtype=int) * label
        labels.append(label_epoch)

        total_duration += duration_sec

    labels = np.hstack(labels)

    # Extract signal from the selected channel
    ch_names = psg_f.getSignalLabels()
    ch_samples = psg_f.getNSamples()
    if isinstance(select_ch, str):
        select_ch = [select_ch]
    select_ch_idx = [-1, -1]
    for ch_pair in select_ch:
        for s in range(psg_f.signals_in_file):
            if ch_names[s].startswith(ch_pair[0]):
                select_ch_idx[0] = s
            if ch_names[s].startswith(ch_pair[1]):
                select_ch_idx[1] = s
        if select_ch_idx[0] >= 0 and select_ch_idx[1] >= 0:
            print("Select channels " + ", ".join(ch_pair) + \
                " (" + ", ".join([ch_names[i] for i in select_ch_idx]) + ")")
            break
    if select_ch_idx[0] == -1 or select_ch_idx[1] == -1:
        # raise Exception("Channel not found.")
        select_ch_idx = [select_ch_idx[0]]
        print("Expect channels " + ", ".join(ch_pair) + \
                ", but only found " + ", ".join([ch_names[i] for i in select_ch_idx]))
    select_ch_label = ", ".join([ch_names[i] for i in select_ch_idx])

    if len(select_ch_idx) == 1:
        # raise Exception("Channel pair not found.")
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx[0])
        signals_raw = psg_f.readSignal(select_ch_idx[0])
        if sampling_rate != target_fs:
            signals = resample(signals_raw, sampling_rate, target_fs)
    else:
        sampling_rate = psg_f.getSampleFrequency(select_ch_idx[0])
        signals_raw_0 = psg_f.readSignal(select_ch_idx[0])
        signals_raw_1 = psg_f.readSignal(select_ch_idx[1])
        if sampling_rate != target_fs:
            signals_0 = resample(signals_raw_0, sampling_rate, target_fs)
            signals_1 = resample(signals_raw_1, sampling_rate, target_fs)
        signals = signals_0 - signals_1

    signals_epoched = []
    for i in range(len(labels)): # no more than the number of labels
        epoch_begin = int(round((ann_onsets[i]-overlap_left)*target_fs))
        epoch_begin = max(0, epoch_begin) # workaround for starting in less than 5 sec
        epoch_end = epoch_begin + int(round(epoch_duration*target_fs))
        if epoch_end > len(signals): # workaround for ending in less than 5 sec
            epoch_end = len(signals)
            epoch_begin = epoch_end - int(round(epoch_duration*target_fs))
        signal_epoch = signals[epoch_begin:epoch_end]
        signals_epoched.append(signal_epoch)
    signals = np.vstack(signals_epoched)

    # Sanity check
    assert len(signals) == len(labels), f"signal: {signals.shape} != {len(labels)}"
    n_epochs = len(signals)

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)
    y_full = y.copy()
    keep_idx = np.arange(len(y_full))

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
        "ch_label": select_ch_label,
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
    return data, labels, sampling_rate

def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    fs = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
        if fs is None:
            fs = sampling_rate
        elif fs != sampling_rate:
            raise Exception("Found mismatch in sampling rate.")

        # Reshape the data to match the input of the model - conv2d
        # tmp_data = np.squeeze(tmp_data)
        # tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
        
        # # Reshape the data to match the input of the model - conv1d
        tmp_data = tmp_data[:, :, np.newaxis]

        # Casting
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)

        data.append(tmp_data)
        labels.append(tmp_labels)

    return data, labels

def load_subdata_preprocessed(datapath, subject):
    npz_f = os.path.join(datapath, subject+'.npz')
    data, labels, fs = load_npz_file(npz_f)
    return data, labels

def load_dataset_preprocessed(datapath, subsets=['MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5']):
    if isinstance(subsets, str):
        subsets = [subsets]
    npzfiles = []
    for subset in subsets:
        subset_npzfiles = glob.glob(os.path.join(datapath, subset, "*.npz"))
        [npzfiles.append(npz_f) for npz_f in subset_npzfiles]
    npzfiles.sort()
    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]
    data, labels = load_npz_list_files(npzfiles)
    return data, labels, subjects

def load_npzlist_preprocessed(datapath, subsets=['MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5']):
    if isinstance(subsets, str):
        subsets = [subsets]
    npzfiles = []
    for subset in subsets:
        subset_npzfiles = glob.glob(os.path.join(datapath, subset, "*.npz"))
        [npzfiles.append(npz_f) for npz_f in subset_npzfiles]
    npzfiles.sort()
    return npzfiles


if __name__ == "__main__":

    datapath = '/home/public/datasets/eegdata/sleep/mass/'
    savepath = '/home/yuty2009/data/eegdata/sleep/mass/processed/'
    os.makedirs(savepath, exist_ok=True)
    
    subsets = ['MASS1', 'MASS2', 'MASS3', 'MASS4', 'MASS5']
    select_ch = [
        ['EEG C3', 'EEG A2'], # C3-A2
        ['EEG C4', 'EEG A1'], # C4-A1
        ['EEG C3', 'EEG A2'], # use only C3 otherwise
    ] 

    annotations = []
    ann_f = open(savepath+'/annotations.txt', 'w')
    for subset in subsets:
        psg_fnames = glob.glob(os.path.join(datapath, subset, "*PSG.edf"))
        ann_fnames = glob.glob(os.path.join(datapath, subset, "*Base.edf"))
        psg_fnames.sort()
        ann_fnames.sort()

        os.makedirs(savepath+subset, exist_ok=True)

        for i in range(len(psg_fnames)):

            subject = os.path.basename(psg_fnames[i])[:-8]

            print(f"Load and extract continuous EEG into epochs for subset {subset} subject {subject}")
            data_dict = load_eegdata_mass(psg_fnames[i], ann_fnames[i], select_ch=select_ch)

            annotations.append(data_dict["y_full"])
            ann_f.write(",".join([f"{ann}" for ann in data_dict["y_full"]]) + "\n")

            np.savez(savepath+subset+'/'+subject+'.npz', **data_dict)
    ann_f.close()