import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib

def ScatterHighDimensionalData(X, Classes, UseTSNE=False, LowDim=2, gax=-1):
    Projector = TSNE(n_components=LowDim) if UseTSNE else PCA(n_components=LowDim)
    if gax == -1:
        fig = plt.figure()
        ax = fig.gca()
    else:
        ax = gax

    if LowDim == X.shape[1]:
        ProjectedVecs = X
    else:
        ProjectedVecs = Projector.fit_transform(X)
    if LowDim == 1:
        ax.scatter(Classes, ProjectedVecs, s=10)
    elif LowDim == 2:
        ax.scatter(ProjectedVecs[:, 0], ProjectedVecs[:, 1], s=50, c=Classes)
    else:
        if gax == -1:
            ax = fig.add_subplot(projection='3d')
        ax.scatter(ProjectedVecs[:, 0], ProjectedVecs[:, 1], ProjectedVecs[:, 2], c=Classes)
        # ax.axis('off')

def GetMetricsPer1DSig(x,fetal_period,maternal_period,peaks_range):
    fine_tuned_fetal_period = fetal_period - peaks_range + np.argmax(x[fetal_period - peaks_range:fetal_period + peaks_range])
    fine_tuned_maternal_period = maternal_period - peaks_range + np.argmax(x[maternal_period - peaks_range:maternal_period + peaks_range])

    fpeak=x[int(fetal_period)+np.arange(-peaks_range,peaks_range)].max()
    mpeak=x[int(maternal_period)+np.arange(-peaks_range,peaks_range)].max()
    fmdiff=fpeak-mpeak
    nfpeak=fpeak/np.std(x)
    nmpeak=mpeak/np.std(x)
    nfmdiff=fmdiff/np.std(x)
    fpeakdist= np.abs(fine_tuned_fetal_period-fetal_period)

    Metrics={'fpeak':fpeak,'mpeak':mpeak,'fmdiff':fmdiff,'nfpeak':nfpeak,'nmpeak':nmpeak,'nfmdiff':nfmdiff,'fpeakdist':fpeakdist}
    return Metrics

def PlotAndGetRatios(SignalToInspect,Y_F,Y_M,Title=""):
    LowDim=3
    peaks_range=25 #(acctually *2)
    nSpectralInds = min(SignalToInspect.shape[1], LowDim)

    fs = 25
    fig = plt.figure(figsize=(13, 14))
    gs = gridspec.GridSpec(nSpectralInds, 4)

    WeightedMetricsArray={}
    PrincipalMetricsArray={}
    MaxMetricsArray={}

    for b_ind in range(SignalToInspect.shape[0]):
        mat_to_inspect = SignalToInspect[b_ind, :, :].detach().cpu().T
        fetal_period = int(Y_F[b_ind].item())
        maternal_period = int(Y_M[b_ind].item())

        if mat_to_inspect.shape[-1] > 1:
            LowDimToUse = min(mat_to_inspect.shape[-1], LowDim)
            Projector = PCA(n_components=LowDimToUse)
            ProjectedVecs = Projector.fit_transform(mat_to_inspect)
            Eigenvals = Projector.explained_variance_.ravel()
        else:
            Projector =  PCA(n_components=1)
            ProjectedVecs = Projector.fit_transform(mat_to_inspect)
            Eigenvals =  Projector.explained_variance_.ravel()
        Weights = Eigenvals / sum(Eigenvals)
        WeightedMetrics={}
        PrincipalMetrics={}
        MaxMetrics={}
        for spectral_ind in range(nSpectralInds):
            x = ProjectedVecs[:, spectral_ind]
            x = x - np.mean(x)
            x = x / np.linalg.norm(x)
            x = sc.signal.correlate(x, x, mode='full')[len(x) - 1:]
            # x = x - np.mean(x)
            # x = x / np.linalg.norm(x)

            fine_tuned_fetal_period = fetal_period - peaks_range + np.argmax(x[fetal_period - peaks_range:fetal_period + peaks_range])
            fine_tuned_maternal_period = maternal_period - peaks_range + np.argmax(x[maternal_period - peaks_range:maternal_period + peaks_range])
            fm_ratio = x[fine_tuned_fetal_period] / x[fine_tuned_maternal_period]

            if b_ind<4:
                ax = fig.add_subplot(gs[spectral_ind, b_ind])
                ax.plot(x)
                ax.scatter([fine_tuned_maternal_period], x[fine_tuned_maternal_period],s=100)
                ax.scatter([fine_tuned_fetal_period], x[fine_tuned_fetal_period],s=100)
                ax.set_xlim([0, 1000])
                ax.set_title('Sample #%g \n ratio=%.3g, weight=%.3g' % (b_ind, fm_ratio, Weights[spectral_ind]))

            peaks_dist=np.abs(fine_tuned_fetal_period-fetal_period)
            Metrics=GetMetricsPer1DSig(x,fetal_period,maternal_period,peaks_range)
            for k in list(Metrics.keys()):
                if not(k in WeightedMetrics):
                    WeightedMetrics[k]=0
                WeightedMetrics[k]+=Metrics[k]*Weights[spectral_ind]
                if spectral_ind==0:
                    PrincipalMetrics[k]=Metrics[k]
                    MaxMetrics[k]=Metrics[k]
                else:
                    if Metrics[k]>MaxMetrics[k]:
                        MaxMetrics[k]=Metrics[k]

        for k in list(Metrics.keys()):
            if not(k in MaxMetricsArray): #for the first batch
                WeightedMetricsArray[k]=[]
                PrincipalMetricsArray[k]=[]
                MaxMetricsArray[k]=[]
            WeightedMetricsArray[k].append(WeightedMetrics[k])
            PrincipalMetricsArray[k].append(PrincipalMetrics[k])
            MaxMetricsArray[k].append(MaxMetrics[k])

    fig.suptitle(Title)
    return fig,WeightedMetricsArray,PrincipalMetricsArray,MaxMetricsArray

def GetMetricsForCode(SignalToInspect,Y_F,Y_M):
    LowDim=3
    peaks_range=25 #(acctually *2)
    nSpectralInds = min(SignalToInspect.shape[1], LowDim)

    WeightedMetricsArray={}
    PrincipalMetricsArray={}
    MaxMetricsArray={}
    for b_ind in range(SignalToInspect.shape[0]):

        mat_to_inspect = SignalToInspect[b_ind, :, :].detach().cpu().T
        fetal_period = int(Y_F[b_ind].item())
        maternal_period = int(Y_M[b_ind].item())

        if mat_to_inspect.shape[-1] > 1:
            LowDimToUse = min(mat_to_inspect.shape[-1], LowDim)
            Projector = PCA(n_components=LowDimToUse)
            ProjectedVecs = Projector.fit_transform(mat_to_inspect)
            Eigenvals = Projector.explained_variance_.ravel()
        else:
            Projector =  PCA(n_components=1)
            ProjectedVecs = Projector.fit_transform(mat_to_inspect)
            Eigenvals =  Projector.explained_variance_.ravel()
        Weights = Eigenvals / sum(Eigenvals)
        WeightedMetrics={}
        PrincipalMetrics={}
        MaxMetrics={}
        for spectral_ind in range(nSpectralInds):
            x = ProjectedVecs[:, spectral_ind]
            x = x - np.mean(x)
            x = x / np.linalg.norm(x)
            x = sc.signal.correlate(x, x, mode='full')[len(x) - 1:]
            Metrics=GetMetricsPer1DSig(x,fetal_period,maternal_period,peaks_range)
            for k in list(Metrics.keys()):
                if not(k in WeightedMetrics):
                    WeightedMetrics[k]=0
                WeightedMetrics[k]+=Metrics[k]*Weights[spectral_ind]
                if spectral_ind==0:
                    PrincipalMetrics[k]=Metrics[k]
                    MaxMetrics[k]=Metrics[k]
                else:
                    if Metrics[k]>MaxMetrics[k]:
                        MaxMetrics[k]=Metrics[k]
        for k in list(Metrics.keys()):
            if not(k in MaxMetricsArray): #for the first batch
                WeightedMetricsArray[k]=[]
                PrincipalMetricsArray[k]=[]
                MaxMetricsArray[k]=[]
            WeightedMetricsArray[k].append(WeightedMetrics[k])
            PrincipalMetricsArray[k].append(PrincipalMetrics[k])
            MaxMetricsArray[k].append(MaxMetrics[k])
    return WeightedMetricsArray,PrincipalMetricsArray,MaxMetricsArray

def GetScores(MetricsArray):
    Score_Mean,Score_Std={},{}
    Score_Mean['Ratio']= np.mean(MetricsArray['fpeak'])/ np.mean(MetricsArray['mpeak'])
    Score_Mean['Diff']= np.mean(np.array(MetricsArray['fpeak'])-np.array(MetricsArray['mpeak']))
    Score_Std['Ratio']= np.std(MetricsArray['fpeak'])/ np.std(MetricsArray['mpeak'])
    Score_Std['Diff']= np.std(np.array(MetricsArray['fpeak'])-np.array(MetricsArray['mpeak']))
    return Score_Mean,Score_Std
