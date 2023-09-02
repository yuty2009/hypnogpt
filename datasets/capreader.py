
import os
import time
import glob
import mne
import numpy as np
from scipy import interpolate


# Have to manually define based on the dataset
ann2label = {
    "W": 0,
    "S1": 1,
    "S2": 2,
    "S3": 3, "S4": 3, # Follow AASM Manual
    "REM": 4, "R": 4,
    "MT": 5,
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
    "S1": N1,
    "S2": N2,
    "S3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNKNOWN": UNKNOWN,
}

disorder_prefix = {
    "brux": 1,
    "ins": 2,
    "narco": 3,
    "nfle": 4,
    "plm": 5,
    "rbd": 6,
    "sdb": 7,
    "n": 0,
}


def get_timeStamp(str):
    try:
        timeArray = time.strptime(str, "%H:%M:%S")
    except:
        timeArray = time.strptime(str, "%H.%M.%S")
    return timeArray.tm_hour*3600 + timeArray.tm_min*60 + timeArray.tm_sec


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


def load_annotations(ann_fname, interval_time=30):
    f = open(ann_fname, 'r')
    lines = f.readlines()
    f.close()
    start_index = 9999
    style = 0
    temp_stage = []
    temp_time = []

    start_time = 0
    for i in range(len(lines)):
        if i < start_index:
            if lines[i] == "Sleep Stage	Time [hh:mm:ss]	Event	Duration[s]	Location\n":
                start_index = i
                style = 1
            elif lines[i] == "Sleep Stage	Position	Time [hh:mm:ss]	Event	Duration[s]	Location\n":
                start_index = i
                style = 2
            elif lines[i] == "Sleep Stage	Position	Time [hh:mm:ss]	Event	Duration [s]	Location\n":
                start_index = i
                style = 2
                print ("in")
            else:
                continue

        if i == start_index + 1:
            temp = lines[i].split("\t")
            start_time = get_timeStamp(temp[style])
            
        if i > (interval_time / 30 - 1) / 2 + start_index and i + (interval_time / 30 - 1) / 2 < len(lines):
            if style == 0:
                print(f"{ann_fname} time column error")
                return 1, 1
            
            temp = lines[i].split("\t")
            try:
                if temp[style+2] != '30':
                    continue
                timeStamp = get_timeStamp(temp[style])
                temp_time.append(timeStamp)
                temp_stage.append(temp[0])
            except:
                if i - start_index > 3:
                    print(f"{ann_fname} duration warning")
                    break
                else:
                    print(f"{ann_fname} duration error")
                    exit()
    times = np.array(temp_time)
    if times.shape[0] == 0:
        print(f"{ann_fname} load txt error")
        return 1, 1

    return temp_stage, times


def load_eegdata_cap(psg_fname, ann_fname, select_ch=['C4-A1', 'C4A1', 'C3-A2', 'C3A2'], target_fs=100.):
    """
    https://github.com/emadeldeen24/AttnSleep
    """

    labels, times = load_annotations(ann_fname)
    if labels == 1 and times == 1:
        print(f"load annotations from {ann_fname} failed")
        return
    labels = np.array([ann2label[x] for x in labels])

    data = mne.io.read_raw_edf(psg_fname)
    sampling_rate = data.info['sfreq']
    try:
        signals_raw = data.get_data(picks=select_ch[0])[0]
    except:
        try:
            signals_raw = data.get_data(picks=select_ch[1])[0]
        except:
            try:
                signals_raw = data.get_data(picks=select_ch[2])[0]
            except:
                try:
                    signals_raw = data.get_data(picks=select_ch[3])[0]
                except:
                    try:
                        signals_0 = data.get_data(picks='A1')[0]
                        signals_1 = data.get_data(picks='C4')[0]
                        signals_raw = signals_1 - signals_0
                    except:
                        print(f"{psg_fname} does not have channel {select_ch[0]} or {select_ch[2]}")
                        return
    signals = signals_raw
    if sampling_rate != target_fs:
        signals = resample(signals_raw, sampling_rate, target_fs)

    #### dealing time ##########
    #timeArray = time.strptime(data.info['meas_date'], "%Y-%m-%d %H:%M:%S+00:00")
    #start_time = timeArray.tm_hour*3600 + timeArray.tm_min*60 + timeArray.tm_sec
    start_time = data.info['meas_date'].hour * 3600 + data.info['meas_date'].minute * 60 + data.info['meas_date'].second
    for i in range(len(times)):
        if times[i] >= start_time:
            times[i] = times[i] - start_time
        else:
            times[i] = 24*3600 - start_time + times[i]

    signals_epoched = []
    for i in range(len(times)):
        epoch_begin = int(times[i] * target_fs)
        epoch_end = int((times[i] + 30) * target_fs)
        if epoch_end > len(signals):
            print(f"{psg_fname} time overflow, rest epoch {len(times) - i}\n")
            break
        signal_epoch = signals[epoch_begin:epoch_end]
        signals_epoched.append(signal_epoch)
    signals = np.vstack(signals_epoched)

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)

    print(x.shape)
    print(y.shape)
    # assert len(x) == len(y)
    y_full = y.copy()
    y = y[:len(x)]

    # Remove movement and unknown
    remove_idx = np.where(y >= 5)[0]
    if len(remove_idx) > 0:
        select_idx = np.setdiff1d(np.arange(len(x)), remove_idx)
        x = x[select_idx]
        y = y[select_idx]

    remove_idx = np.where(y_full >= 5)[0]
    if len(remove_idx) > 0:
        select_idx = np.setdiff1d(np.arange(len(y_full)), remove_idx)
        y_full = y_full[select_idx]

    data_dict = {
        "x": x,
        "y": y,
        "y_full": y_full,
        "fs": sampling_rate
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

def load_dataset_preprocessed(datapath, subsets=['shhs1'], n_subjects=None):
    if isinstance(subsets, str):
        subsets = [subsets]
    npzfiles = []
    for subset in subsets:
        subset_npzfiles = glob.glob(os.path.join(datapath, subset, "*.npz"))
        [npzfiles.append(npz_f) for npz_f in subset_npzfiles]
    npzfiles.sort()
    if n_subjects is not None:
        npzfiles = npzfiles[:n_subjects]
    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]
    data, labels = load_npz_list_files(npzfiles)
    return data, labels, subjects


if __name__ == '__main__':

    datapath = 'e:/eegdata/sleep/cap/'
    savepath = datapath + 'processed/'
    os.makedirs(savepath, exist_ok=True)

    record = open(datapath + 'RECORDS', 'r')
    lines = record.readlines()
    
    annotations = []
    ann_f = open(savepath+'annotations.txt', 'w')
    sub_f = open(savepath+'subject_labels.txt', 'w')
    for i in range(len(lines)):
        temp = lines[i].split('.')
        ann_fname = datapath + temp[0] + '.txt'
        edf_fname = datapath + temp[0] + '.edf'

        subject = os.path.basename(edf_fname)[:-4]
        for prefix, label in disorder_prefix.items():
            if subject.startswith(prefix):
                subject_label = label
                break
        sub_f.write(f"{subject_label}\n")
        
        print('Load and extract continuous EEG into epochs for subject '+subject)
        data_dict = load_eegdata_cap(edf_fname, ann_fname)
        annotations.append(data_dict["y_full"])
        ann_f.write(",".join([f"{ann}" for ann in data_dict["y_full"]]) + "\n")

        np.savez(savepath+subject+'.npz', **data_dict)
    ann_f.close()
    sub_f.close()
