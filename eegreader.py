
import torch
import random
import numpy as np
import scipy.signal as signal


class ToTensor(object):
    """ 
    Turn a (timepoints x channels) or (T, C) epoch into 
    a (depth x timepoints x channels) or (D, T, C) image for torch.nn.Convnd
    """
    def __call__(self, epoch, target=None):
        if isinstance(epoch, np.ndarray):
            epoch = torch.FloatTensor(epoch.copy()[None, :, :])
        elif isinstance(epoch, torch.Tensor):
            epoch = epoch[None, :, :]
        if target is not None:
            return epoch, torch.LongTensor(target)
        return epoch
    

class TransformSTFT(object):
    """ Translate epoch data (t, c) into grayscale images (1, N, T, c)
    # N is the number of frequencies where STFT is applied and 
    # T is the total number of frames used.
    """
    def __call__(self, epoch):
        epoch = torch.FloatTensor(epoch)
        epoch_spectra = []
        channels = epoch.size(-1)
        for i in range(channels):
            spectra = torch.stft(
                epoch[..., i], 
                n_fft=256, 
                hop_length=100, 
                win_length=200,
                window=torch.hamming_window(200),
                center=False,
                onesided=True,
                return_complex=False,
            )
            spectra_real = spectra[..., 0]
            spectra_imag = spectra[..., 1]
            spectra_magn = torch.abs(torch.sqrt(torch.pow(spectra_real,2)+torch.pow(spectra_imag,2)))
            spectra_magn = 20*torch.log10(spectra_magn)
            epoch_spectra.append(spectra_magn.unsqueeze(dim=-1))
        epoch_spectra = torch.cat(epoch_spectra, dim=-1)
        if channels <= 1:
            epoch_spectra = epoch_spectra.squeeze(-1)
        return epoch_spectra.unsqueeze(dim=0)
    

class TransformFilterBank(object):
    """ Translate epoch data (t, c) into grayscale images (1, N, t, c)
    # N is the number of frequency bands
    # T is the total number of frames used.
    """
    def __init__(self, filtbanks, fs, filttype='filter'):
        self.fs = fs
        self.filttype = filttype
        self.filtbanks = filtbanks

    def __call__(self, epoch):
        epoch_filtered = []
        # repetitively filter the data.
        for filtband in self.filtbanks:
            epoch_band = np.zeros_like(epoch)
            for i in range(epoch.shape[-1]):
                epoch_band[:, i] = self.bandpassFilter(epoch[:, i], filtband, self.fs, 2, -1, self.filttype)
            epoch_band = epoch_band[np.newaxis, :, :]
            epoch_filtered.append(torch.from_numpy(epoch_band).float())
        epoch_filtered = torch.cat(epoch_filtered, dim=0)
        # squeeze the last dimension if only there is only one channel
        return epoch_filtered.unsqueeze(dim=0)

    def bandpassFilter(self, data, bandFiltCutF, fs, filtAllowance=2, axis=1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30 # stopband attenuation
        aPass = 3 # passband attenuation
        nFreq= fs/2 # Nyquist frequency
        
        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data
        
        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass =  bandFiltCutF[1]/ nFreq
            fStop =  (bandFiltCutF[1]+filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
        
        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass =  bandFiltCutF[0]/ nFreq
            fStop =  (bandFiltCutF[0]-filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')
        
        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass =  (np.array(bandFiltCutF)/ nFreq).tolist()
            fStop =  [(bandFiltCutF[0]-filtAllowance)/ nFreq, (bandFiltCutF[1]+filtAllowance)/ nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, epochs, targets, transforms=None):
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.targets = torch.LongTensor(targets)
        self.classes = {'w' : 0, '1' : 1, '2' : 2, '3' : 3, 'R' : 4}

    def __getitem__(self, idx):
        return self.epochs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
    

class SubEEGDataset(torch.utils.data.Dataset):
    def __init__(self, npzfile, transforms=None):
        with np.load(npzfile) as f:
            epochs = f["x"][:, :, np.newaxis] # for conv2d
            targets = f["y"]
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.targets = torch.LongTensor(targets)
        self.classes = {'w' : 0, '1' : 1, '2' : 2, '3' : 3, 'R' : 4}

    def __getitem__(self, idx):
        return self.epochs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
    

class BigEEGDataset(torch.utils.data.Dataset):
    def __init__(self, npzfiles, transforms=None):
        self.npzfiles = npzfiles
        random.shuffle(self.npzfiles) # use on-demand
        self.transforms = transforms
        self.data_gen = self.get_data()
        self.classes = {'w' : 0, '1' : 1, '2' : 2, '3' : 3, 'R' : 4}

    def get_data(self):
        for npzfile in self.npzfiles:
            with np.load(npzfile) as f:
                self.epochs = f["x"][:, :, np.newaxis] # for conv2d
                self.targets = f["y"]
            if self.transforms != None:
                self.epochs = [self.transforms(epoch) for epoch in self.epochs]
            self.targets = torch.LongTensor(self.targets)
            if len(self.targets) > 0:
                yield (self.epochs.pop(), self.targets.pop())

    def __getitem__(self, idx):
        return next(self.data_gen)

    def __len__(self):
        return len(self.npzfiles)*3000


class SeqEEGDataset(torch.utils.data.Dataset):
    def __init__(self, epochs, targets, seqlen, transforms=None):
        if transforms == None:
            self.epochs = torch.FloatTensor(epochs)
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
            self.epochs = torch.stack(self.epochs)

        self.targets = torch.LongTensor(targets)
        self.seqlen = seqlen
        self.classes = {'w' : 0, '1' : 1, '2' : 2, '3' : 3, 'R' : 4}
        assert self.seqlen <= len(self), "seqlen is too large"

    def __getitem__(self, idx):
        epoch_seq = torch.zeros(
            (self.seqlen,)+self.epochs.shape[1:], 
            dtype=self.epochs.dtype,
            device=self.epochs.device
            )
        idx1 = idx + 1
        if idx1 < self.seqlen:
            epoch_seq[-idx1:] = self.epochs[:idx1]
        else:
            epoch_seq = self.epochs[idx1-self.seqlen:idx1]
        return epoch_seq, self.targets[idx]

    def __len__(self):
        return len(self.targets) 
    

class SubSeqEEGDataset(torch.utils.data.Dataset):
    def __init__(self, npzfile, seqlen, transforms=None):
        with np.load(npzfile) as f:
            epochs = f["x"][:, :, np.newaxis] # for conv2d
            targets = f["y"]
        if transforms == None:
            self.epochs = torch.FloatTensor(epochs)
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
            self.epochs = torch.stack(self.epochs)

        self.targets = torch.LongTensor(targets)
        self.seqlen = seqlen
        self.classes = {'w' : 0, '1' : 1, '2' : 2, '3' : 3, 'R' : 4}
        assert self.seqlen <= len(self), "seqlen is too large"

    def __getitem__(self, idx):
        epoch_seq = torch.zeros(
            (self.seqlen,)+self.epochs.shape[1:], 
            dtype=self.epochs.dtype,
            device=self.epochs.device
            )
        idx1 = idx + 1
        if idx1 < self.seqlen:
            epoch_seq[-idx1:] = self.epochs[:idx1]
        else:
            epoch_seq = self.epochs[idx1-self.seqlen:idx1]
        return epoch_seq, self.targets[idx]

    def __len__(self):
        return len(self.targets) 
    

if __name__ == '__main__':

    x = torch.rand(3000, 1)

    tf_stft = TransformSTFT()
    tf_filtbank = TransformFilterBank(
        [[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]],
        fs = 250,
    )

    x_spectra = tf_stft(x)
    print(x_spectra.shape)

    x_filtbank = tf_filtbank(x)
    print(x_filtbank.shape)
