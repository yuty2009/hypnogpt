
import os
import glob
import pyedflib
import numpy as np
import pandas as pd
from scipy import interpolate
import xml.etree.ElementTree as ET


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


def load_eegdata_shhs(edf_fname, ann_fname, select_ch='EEG', target_fs=100.):
    """
    https://github.com/emadeldeen24/AttnSleep
    """

    labels = []
    # Read annotation and its header
    t = ET.parse(ann_fname)
    r = t.getroot()
    for i in range(len(r[4])):
        lbl = int(r[4][i].text)
        if lbl == 4:  # make stages N3, N4 same as N3
            labels.append(3)
        elif lbl == 5:  # Assign label 4 for REM stage
            labels.append(4)
        else:
            labels.append(lbl)
        if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
            # print( "============================== Faulty file ==================")
            # return 
            pass
    labels = np.asarray(labels)

    edf_f = pyedflib.EdfReader(edf_fname)

    # assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
    start_datetime = edf_f.getStartdatetime()

    file_duration = edf_f.getFileDuration()
    epoch_duration = 30.0 # psg_f.datarecord_duration

    # Extract signal from the selected channel
    ch_names = edf_f.getSignalLabels()
    ch_samples = edf_f.getNSamples()
    select_ch_idx = -1
    for s in range(edf_f.signals_in_file):
        if ch_names[s].startswith(select_ch):
            select_ch_idx = s
            break
    if select_ch_idx == -1:
        raise Exception("Channel not found.")
    sampling_rate = edf_f.getSampleFrequency(select_ch_idx)
    signals = edf_f.readSignal(select_ch_idx)
    if sampling_rate != target_fs:
        signals = resample(signals, sampling_rate, target_fs)
    n_epoch_samples = int(epoch_duration * target_fs)
    signals = signals.reshape(-1, n_epoch_samples)

    # Get epochs and their corresponding labels
    x = signals.astype(np.float32)
    y = labels.astype(np.int32)
    y_full = y.copy()
    keep_idx = np.arange(len(y_full))

    print(x.shape)
    print(y.shape)
    assert len(x) == len(y)

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

    data_dict = {
        "x": x,
        "y": y,
        "y_full": y_full,
        "keep_idx": keep_idx,
        "fs": target_fs,
        "ch_label": select_ch,
        "edf_path": edf_fname,
    }

    return data_dict

def load_variables_shhs(
        var_fname, 
        var_target = 'visitnumber',
        target_value = 1,
        var_list=['nsrrid', 'nsrr_ahi_hp3u', 'nsrr_ahi_hp3r_aasm15', 'nsrr_ahi_hp4u_aasm15', 'nsrr_ahi_hp4r']
    ):
    df = pd.read_csv(var_fname)
    var_matrics = df[var_list].values
    var_target = var_matrics[df[var_target] == target_value]
    return var_target

def load_npz_file(npz_file):
    """Load data and labels from a npz file."""
    with np.load(npz_file) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
        ch_label = f["ch_label"]
        keep_idx = f['keep_idx']
        edf_path = f['edf_path']
    return data, labels, sampling_rate, ch_label, keep_idx, edf_path

def load_npz_list_files(npz_files):
    """Load data and labels from list of npz files."""
    data = []
    labels = []
    ch_labels = []
    keep_idxs = []
    edf_paths = []
    fs = None
    for npz_f in npz_files:
        print("Loading {} ...".format(npz_f))
        tmp_data, tmp_labels, sampling_rate, ch_label, keep_idx, edf_path = load_npz_file(npz_f)
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
        ch_labels.append(ch_label)
        keep_idxs.append(keep_idx)
        edf_paths.append(edf_path)

    return data, labels, ch_labels, keep_idxs, edf_paths

def load_subdata_preprocessed(datapath, subject):
    npz_f = os.path.join(datapath, subject+'.npz')
    data, labels, fs, _, _, _ = load_npz_file(npz_f)
    return data, labels

def load_dataset_preprocessed(datapath, n_subjects=None, extra_info=False):
    npzfiles = glob.glob(os.path.join(datapath, "*.npz"))
    npzfiles.sort()
    if n_subjects is not None:
        npzfiles = npzfiles[:n_subjects]
    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]
    data, labels, ch_labels, keep_idxs, edf_paths = load_npz_list_files(npzfiles)
    if extra_info:
        return data, labels, subjects, ch_labels, keep_idxs, edf_paths
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

    # datapath = 'e:/eegdata/sleep/shhs/'
    datapath = '/home/yuty2009/data/eegdata/sleep/shhs/'
    edf_dir = os.path.join(datapath, 'polysomnography/edfs/shhs1/')
    ann_dir = os.path.join(datapath, 'polysomnography/annotations-events-profusion/shhs1/')
    var_fname = os.path.join(datapath, 'datasets/shhs-harmonized-dataset-0.20.0.csv')

    savepath = os.path.join(datapath, 'processed/shhs1_329/')
    if not os.path.isdir(savepath):
        os.makedirs(savepath, exist_ok=True)

    var_matrics = load_variables_shhs(var_fname)

    ## Load selected 329 files
    select_filepath = os.path.join(datapath, 'selected_shhs1_files.txt')
    ids = pd.read_csv(select_filepath, header=None, names=['a'])
    ids = ids['a'].values.tolist()
    ## Load all files
    # edf_fnames = glob.glob(os.path.join(edf_dir, "*.edf"))
    # ids = []
    # data_ahi = []
    # for edf_f in edf_fnames:
    #     subid = os.path.basename(edf_f)[:-4]
    #     nsrrid = subid.split('-')[1]
    #     ids.append(subid)
    #     data_ahi.append(var_matrics[var_matrics[:, 0] == int(nsrrid)])
    # data_ahi = np.concatenate(data_ahi, axis=0)
    # np.savez(savepath+'ahi.npz', data_ahi=data_ahi)

    edf_fnames = [os.path.join(edf_dir, i + ".edf") for i in ids]
    ann_fnames = [os.path.join(ann_dir,  i + "-profusion.xml") for i in ids]

    edf_fnames.sort()
    ann_fnames.sort()

    annotations = []
    ann_f = open(savepath+'annotations.txt', 'w')
    for i in range(len(edf_fnames)):
        subject = os.path.basename(edf_fnames[i])[:-4]
        print('Load and extract continuous EEG into epochs for subject '+subject)
        data_dict = load_eegdata_shhs(edf_fnames[i], ann_fnames[i])
        annotations.append(data_dict["y_full"])
        ann_f.write(",".join([f"{ann}" for ann in data_dict["y_full"]]) + "\n")
        np.savez(savepath+subject+'.npz', **data_dict)
    ann_f.close()
