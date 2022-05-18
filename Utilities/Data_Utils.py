#TODOS
# PlotACChannel
# ShowSample

##
import os
from scipy import signal
from scipy.signal import medfilt
import numpy as np
import torch
import scipy as sc
from scipy import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
np.random.seed(42)
torch.manual_seed(42)

##
def PlotACChannel(x,ax,channel,instaneous_fhr,instaneous_mhr):

    fetal_period = int(instaneous_fhr)
    maternal_period = int(instaneous_mhr)

    x = x[channel, :]
    x = x - np.mean(x)
    x = x / np.linalg.norm(x)
    x = sc.signal.correlate(x, x, mode='full')[len(x) - 1:]
    x = x - np.mean(x)
    x = x / np.linalg.norm(x)

    fine_tuned_fetal_period=fetal_period
    fine_tuned_maternal_period=maternal_period
    fm_ratio = np.abs(x[fine_tuned_fetal_period] / x[fine_tuned_maternal_period])
    fm_gain = np.abs(x[fine_tuned_fetal_period])-np.abs(x[fine_tuned_maternal_period])

    ax.plot(x)
    ax.scatter([fine_tuned_maternal_period], x[fine_tuned_maternal_period],s=100,label='maternal peak')
    ax.scatter([fine_tuned_fetal_period], x[fine_tuned_fetal_period],s=100,label='fetal peak')
    ax.legend()
    ax.set_xlim([100, 1000])
    ax.set_ylim([min(x[100:1000]),1.1*max(x[100:1000])])
    ax.axis('off')
    return ax,fm_ratio,fm_gain

def EstimateHeartRate(Sig,MinDist,UseAutoCorr=False,nPeaksForHREstimation=5):
    if not(UseAutoCorr):
        peaks, _ = sc.signal.find_peaks(Sig,distance=MinDist)
    else:
        corr=sc.signal.correlate(Sig,Sig,mode='full')[len(Sig)-1:]
        peaks, _ = sc.signal.find_peaks(corr,distance=MinDist)
    return np.median(np.diff(peaks,1)[:nPeaksForHREstimation]),peaks

def PreProcess(input_signal, fs_in=2048, fs_out=500, window_len=int(1 + (0.2 * 500)), band_width=125):
    sig_len=input_signal.shape[0]
    #resample
    resampled_sig_len = signal.resample(input_signal, int(np.floor(sig_len * fs_out / fs_in)), axis=0)
    #remove drift
    sig = resampled_sig_len - medfilt(resampled_sig_len, window_len)
    # apply notch
    b, a = signal.iirnotch(w0=50, Q=4, fs=fs_out)
    sig = signal.filtfilt(b, a, sig)
    #apply LP
    ntaps=1000
    filter_taps = signal.firwin(numtaps=ntaps, cutoff= (2 * band_width) / fs_out, window=('kaiser', 8))
    sig = signal.lfilter(filter_taps, [1], sig)
    sig = sig[int((ntaps + 1) / 2):]
    #apply zscore (not mandatory, just for visualizations)
    sig = sig - np.mean(sig, axis=0)
    sig = sig / np.linalg.norm(sig)

    return sig

class MyDataLoader:
    def __init__(self, DataFolder='Data', Subjects=-1, nChannelsAb=24, nChannelsCh=3, window_len=101, band_width=125, device=torch.device('cpu')):
        self.nChannelsAb = nChannelsAb
        self.nChannelsCh = nChannelsCh
        self.device = device
        self.Subjects = np.arange(1,61) if Subjects == -1 else Subjects
        self.nSubjects = len(self.Subjects)
        self.SubjectsDict = {}

        fs_in=2048
        fs_out = 500
        self.MaxFHR = 160
        self.MaxMHR = 120

        samples_to_chop = 1 * fs_out
        min_sig_len = 4 * fs_out
        nPeaksForHREstimation = 5


        for subject_ind in self.Subjects:
            raw_data = sc.io.loadmat(os.path.join(DataFolder, "%g_E" % subject_ind))['x']
            pre_processed_data=[PreProcess(raw_channel_sig, window_len=window_len, band_width=band_width) for raw_channel_sig in raw_data]
            pre_processed_data = np.vstack(pre_processed_data)


            CurrSubjectDict = {}
            sorted_channel_abs = np.array([2, 1, 17, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23])
            CurrSubjectDict['abSig'] = pre_processed_data[sorted_channel_abs[:nChannelsAb], samples_to_chop:]
            CurrSubjectDict['chSig'] = pre_processed_data[24:(24 + nChannelsCh), samples_to_chop:]
            CurrSubjectDict['dSig'] = sc.io.loadmat(os.path.join(DataFolder, "%g_P" % subject_ind))['envelope'].ravel()
            CurrSubjectDict['dSig'] = signal.resample(CurrSubjectDict['dSig'],int(np.floor(CurrSubjectDict['dSig'].shape[0] * fs_out / fs_in)),axis=0)
            CurrSubjectDict['dSig'] = CurrSubjectDict['dSig'][(1000+samples_to_chop):]

            CurrSubjectDict['sigLen'] = len(CurrSubjectDict['dSig'])
            CurrSubjectDict['fhr'], fpeaks = EstimateHeartRate(CurrSubjectDict['dSig'], MinDist=60 * fs_out / self.MaxFHR,
                                                               UseAutoCorr=True, nPeaksForHREstimation=nPeaksForHREstimation)
            CurrSubjectDict['mhr'], mpeaks = EstimateHeartRate(CurrSubjectDict['chSig'][0],
                                                               MinDist=60 * fs_out / self.MaxMHR, UseAutoCorr=False,
                                                               nPeaksForHREstimation=nPeaksForHREstimation)

            if CurrSubjectDict['sigLen'] > min_sig_len:
                self.SubjectsDict[subject_ind] = CurrSubjectDict

            if not (self.device==torch.device('cpu')):
                for subject_dict in self.SubjectsDict:
                    for key in self.SubjectsDict[subject_dict]:
                        self.SubjectsDict[subject_dict][key] = torch.tensor(self.SubjectsDict[subject_dict][key]).to(self.device).float()

    def GetSample(self, offset_range=-1,seg_len=2000,subject_ind=46,device=torch.device('cpu')):
        sig_len =self.SubjectsDict[subject_ind]['sigLen']
        if np.isscalar(offset_range):
            offset = np.random.randint(0, int(sig_len-seg_len), 1)[0]
        else:
            offset = np.random.randint(offset_range[0], offset_range[1], 1)[0]
        x   =  self.SubjectsDict[subject_ind]['abSig'][:,offset:(offset + seg_len)]
        t   =  self.SubjectsDict[subject_ind]['chSig'][:,offset:(offset + seg_len)]
        d   =  self.SubjectsDict[subject_ind]['dSig'][offset:(offset + seg_len)]
        fhr =  self.SubjectsDict[subject_ind]['fhr']
        mhr =  self.SubjectsDict[subject_ind]['mhr']

        if device==torch.device('cpu') and not(self.device==torch.device('cpu')):
            [x,t,d,fhr,mhr]=[x.detach().cpu().numpy() for x in [x,t,d,fhr,mhr]]
        # [x, t, d, fhr, mhr] = [x.to(device) for x in [x, t, d, fhr, mhr]]


        # return x,t,d,fhr,mhr

        return x, t, d, mhr, fhr

    def ShowSample(self, L=2000,subject_ind=-1,offset_range=-1,ab_channel=1,ch_channel=0):
        if subject_ind==-1:
            subject_ind= np.random.choice(list(self.SubjectsDict.keys()))
        x, m, h,y_m,y_f = self.GetSample(seg_len=L,subject_ind=subject_ind,offset_range=offset_range,device='cpu')

        corr=sc.signal.correlate(h,h,mode='full')[len(h)-1:]
        peaks, _ = sc.signal.find_peaks(corr,distance=60*500/200)
        instaneous_fhr=int((np.median(np.diff(peaks,1)[:5])))
        corr=sc.signal.correlate(m[0],m[0],mode='full')[len(h)-1:]
        peaks, _ = sc.signal.find_peaks(corr,distance=60*500/130)
        instaneous_mhr=int((np.median(np.diff(peaks,1)[:5])))


        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(3, 3)


        ax = fig.add_subplot(gs[0, 0])
        plt.plot(h)
        peaks, _ = sc.signal.find_peaks(h,distance=instaneous_fhr*0.9)
        plt.scatter(peaks,h[peaks],color='red',s=200)
        plt.title('Doppler channel')
        plt.xlim([0,L])

        ax = fig.add_subplot(gs[1, 0])
        plt.plot(m[ch_channel])
        plt.title('Chest channel')
        plt.xlim([0,L])

        ax = fig.add_subplot(gs[2, 0])
        plt.plot(x[ab_channel])
        plt.title('Abdominal channel')
        plt.xlim([0,L])


        ax = fig.add_subplot(gs[:, 1])
        nFrames=8
        frame=int(instaneous_fhr)
        xoffset=0
        yoffset=0
        for ind in range(nFrames):
            yoffset=yoffset-min(x[ab_channel][xoffset:(xoffset+frame)])
            plt.plot(yoffset+x[ab_channel][xoffset:(xoffset+frame)])
            yoffset=yoffset+max(x[ab_channel][xoffset:(xoffset+frame)])
            xoffset=xoffset+frame
        plt.xlim([0,frame])
        plt.title('Time wrapping of the abdominal channel')

        ax = fig.add_subplot(gs[:, 2])
        PlotACChannel(x,ax,ab_channel,instaneous_fhr,instaneous_mhr)
        ax.axis('on')
        ax.set_title('Auto-correlation of the abdominal channel')

        fig.suptitle('Subject=%g\n ab_channel=%g, ch_channel=%g'%(subject_ind,ab_channel,ch_channel),fontsize=25)
        fig.show()

    def GetBatch(self, seg_len=1000, BatchSize=128,offset_range=-1,subject_ind=-1):
        X, M, H,Y_M,Y_F,Y_I = [], [], [],[],[],[]

        for _ in range(BatchSize):
            subject_to_use=np.random.choice(list(self.SubjectsDict.keys())) if subject_ind==-1 else subject_ind
            x, m, h ,y_m,y_f= self.GetSample(seg_len=seg_len,offset_range=offset_range,subject_ind=subject_to_use,device=self.device)
            X.append(x)
            M.append(m)
            H.append(h)
            Y_M.append(y_m)
            Y_F.append(y_f)
            Y_I.append(subject_to_use)
        if self.device==torch.device('cpu'):
            X = torch.tensor(np.stack(X)).float()
            M = torch.tensor(np.array(M)).float()
            H = torch.tensor(np.array(H)).float()
            Y_M= torch.tensor(np.array(Y_M)).float()
            Y_F= torch.tensor(np.array(Y_F)).float()
            Y_I= torch.tensor(np.array(Y_I)).float()
        else:
            X = torch.stack(X)
            M = torch.stack(M)
            H = torch.stack(H)
            Y_M = torch.stack(Y_M)
            Y_F = torch.stack(Y_F)
            Y_I =  torch.tensor(np.array(Y_I)).float()
        return X, M, H,Y_M,Y_F,Y_I

    def GetTrajectory(self,seg_len=100,BatchSize=128,Lag=1,subject_ind=-1):
        subject_to_use = np.random.choice(list(self.SubjectsDict.keys())) if subject_ind == -1 else subject_ind
        X, M, H,Y_M,Y_F,Y_I = [], [], [],[],[],[]
        for offset in np.arange(0,BatchSize*Lag,Lag):
            x, m, h ,y_m,y_f= self.GetSample(seg_len=seg_len,offset_range=[offset,offset+1],subject_ind=subject_to_use,device=self.device)
            X.append(x)
            M.append(m)
            H.append(h)
            Y_M.append(y_m)
            Y_F.append(y_f)
            Y_I.append(subject_to_use)
        if self.device == torch.device('cpu'):
            X = torch.tensor(np.stack(X)).float()
            M = torch.tensor(np.array(M)).float()
            H = torch.tensor(np.array(H)).float()
            Y_M= torch.tensor(np.array(Y_M)).float()
            Y_F= torch.tensor(np.array(Y_F)).float()
            Y_I= torch.tensor(np.array(Y_I)).float()
        else:
            X = torch.stack(X)
            M = torch.stack(M)
            H = torch.stack(H)
            Y_M = torch.stack(Y_M)
            Y_F = torch.stack(Y_F)
            Y_I = torch.tensor(np.array(Y_I)).float()

        return X, M, H,Y_M,Y_F,Y_I

##
if 0:
    device=torch.device('cpu')
    device = torch.device("cuda:%g" % 0 if torch.cuda.is_available() else "cpu")
    data_loader=MyDataLoader(Subjects=[46],device=device)
    data_loader.ShowSample(ab_channel=1,ch_channel=0)
    X,T,D,MHR,FHR,SI=data_loader.GetBatch()
