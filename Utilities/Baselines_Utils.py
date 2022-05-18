import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import padasip as pa
import statsmodels.api as sm
import echotorch.nn as etnn
import pickle
from torch.autograd import Variable
from numpy.random import seed
from Utilities.Hyperparameters_Utils import *
from Utilities.Evaluation_Utils import *
import torch
import os
np.random.seed(42)
torch.manual_seed(42)

##
def ApplyESN(X, T,subject_ind,hp_sub,load_trained):
    esn = etnn.LiESN(
        input_dim=T[0].shape[0],
        hidden_dim=hp_sub.hidden_dim,
        output_dim=X[0].shape[0],
        spectral_radius=hp_sub.spectral_radius,
        learning_algo='inv',
        leaky_rate=hp_sub.leaky_rate,
        # with_bias=True,
        # ridge_param=.0001
        ridge_param=hp_sub.ridge_param
    )
    inputs=T[:,:,:hp_sub.InputSegment].clone().detach().float()
    targets=X[:,:,:hp_sub.InputSegment].clone().detach().float()
    inputs, targets = Variable(inputs.permute([0,2,1])), Variable(targets.permute([0,2,1]))
    if load_trained:
        esn.finalize()
        esn.load_state_dict(torch.load(os.path.join(os.getcwd(),'SavedBaselines','Sub%g_ESN'%subject_ind)))
        esn.eval()
    else:
        esn(inputs, targets)
        esn.finalize()
        torch.save(esn.state_dict(), os.path.join(os.getcwd(),'SavedBaselines','Sub%g_ESN'%subject_ind))
    est_ab_sig=esn(inputs)
    Codes=targets.permute([0,2,1])-est_ab_sig.permute([0,2,1])
    return Codes

def ApplyAdaptive(data_loader,subject_ind,MethodToUse,hp_sub,load_trained):
    if data_loader.device==torch.device('cpu'):
        x=data_loader.SubjectsDict[subject_ind]['abSig']
        m=data_loader.SubjectsDict[subject_ind]['chSig']
    else:
        x=data_loader.SubjectsDict[subject_ind]['abSig'].detach().cpu().numpy()
        m=data_loader.SubjectsDict[subject_ind]['chSig'].detach().cpu().numpy()
    TD_chest_sig=[]
    for ch_ind in range(3):
        TD_chest_sig.append(pa.input_from_history(m[ch_ind,:], hp_sub.nTaps))
    TD_chest_sig=np.hstack(TD_chest_sig)
    ab_sig=x[0:hp_sub.nChannelsAb,(int((hp_sub.nTaps+1)/2)-1):((int((hp_sub.nTaps+1)/2)-1)+TD_chest_sig.shape[0])]

    if MethodToUse=='ADALINE':
        if load_trained:
            f = open( os.path.join(os.getcwd(),'SavedBaselines','Sub%g_mcADALINE'%(subject_ind)), 'rb')
            mc_adasgd=pickle.load(f)
            f.close()
        else:
            mc_adasgd = MultiCahnnelAdaline(n_iter=hp_sub.n_iter, eta=hp_sub.eta, random_state=1)
            mc_adasgd.fit(TD_chest_sig, ab_sig)
            f = open( os.path.join(os.getcwd(),'SavedBaselines','Sub%g_mcADALINE'%(subject_ind)), 'wb')
            pickle.dump(mc_adasgd, f)
            f.close()
        est_ab_sig=mc_adasgd.activation(TD_chest_sig)
    if MethodToUse=='OLS':
        ols_model = sm.OLS(ab_sig.T, TD_chest_sig)
        results = ols_model.fit()
        w=results.params
        est_ab_sig=(TD_chest_sig@w).T
    if MethodToUse in ['LMS','RLS','NLMS']:
        est_ab_sig=[]
        for ind in range(ab_sig.shape[0]):
            if MethodToUse=='RLS':
                f = pa.filters.FilterRLS(n=TD_chest_sig.shape[1], w="random", mu=hp_sub.mu)
            if MethodToUse=='LMS':
                f = pa.filters.FilterLMS(n=TD_chest_sig.shape[1], w="random", mu=hp_sub.mu)
            if MethodToUse=='NLMS':
                f = pa.filters.FilterNLMS(n=TD_chest_sig.shape[1], w="random", mu=hp_sub.mu)
            est_ab_sig_tmp, _, _ = f.run(ab_sig[ind],TD_chest_sig)
            est_ab_sig.append(est_ab_sig_tmp)
        est_ab_sig=np.vstack(est_ab_sig)
    e=ab_sig-est_ab_sig

    offsets=np.random.randint(0,e.shape[1]-hp_sub.InputSegment,hp_sub.Navg)
    Codes=[]
    for offset in offsets:
        Codes.append(e[:,offset:(offset+hp_sub.InputSegment)])
    Codes=torch.tensor(np.stack(Codes))
    return Codes
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        average cost in every epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.

    """
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def _initialize_weights(self, m):
        """ use a _initialize_weights method to initialize weights to zero
            and after initialization set w_initialized to True """
        self.w_ = np.zeros(m + 1)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """This method perform one weight update for one training sample xi and calculate the error
        Since weights update will be used is both fit and partial fit methods,
        it's better to seperate it out to be concise"""
        output = self.net_input(xi)
        error = target - output
        self.w_[0] += self.eta * error
        self.w_[1:] += self.eta * xi.dot(error)
        cost = 0.5 * error ** 2
        return cost

    def _shuffle(self, X, y):
        """Shuffle training data with np random permutation"""
        seq = np.random.permutation(len(y))
        return X[seq], y[seq]

    def fit(self, X, y):

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
class MultiCahnnelAdaline(object):
    def __init__(self,nChannels=24, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.nChannels=nChannels
        self.ModelsPerChannel=[]
        for _ in range(nChannels):
            self.ModelsPerChannel.append(AdalineSGD(n_iter=n_iter, eta=eta, random_state=1))

    def fit(self, X, Y):
        for ind in range(self.nChannels):
            self.ModelsPerChannel[ind].fit(X, Y[ind])

    def activation(self, X):
        y=[]
        for ind in range(self.nChannels):
            y.append(self.ModelsPerChannel[ind].activation(X))
        y=np.vstack(y)
        return y

##
def GetScoresForBaeline(MethodToUse,data_loader,subject_ind,load_trained,LogToWandb):
    hp_sub=BaselinesHP()
    hp_sub.MethodToUse=MethodToUse
    X, T, H,Y_M,Y_F,Y_I= data_loader.GetBatch(BatchSize=hp_sub.Navg,seg_len=hp_sub.InputSegment+hp_sub.nTaps-1)
    [X,T,H]=[x.detach().cpu() for x in [X,T,H]]

    if hp_sub.MethodToUse=='ESN':
        hp_sub.nTaps=5
        Codes=ApplyESN(X, T,subject_ind, hp_sub,load_trained)
    else:
        if hp_sub.MethodToUse=='RLS':
            hp_sub.nTaps=5
            hp_sub.mu=0.1
        if hp_sub.MethodToUse=='LMS':
            hp_sub.nTaps=5
            hp_sub.mu=100
        if hp_sub.MethodToUse=='ADALINE':
            hp_sub.nTaps=100
            hp_sub.eta=1
        if hp_sub.MethodToUse=='OLS':
            hp_sub.nTaps=5
        Codes=ApplyAdaptive(data_loader,subject_ind,hp_sub.MethodToUse,hp_sub,load_trained)

    if LogToWandb:
        fig_code,code_WeightedMetricsArray,code_PrincipalMetricsArray,code_MaxMetricsArray = PlotAndGetRatios(Codes,Y_F,Y_M,MethodToUse)
    else:
        code_WeightedMetricsArray,code_PrincipalMetricsArray,code_MaxMetricsArray = GetMetricsForCode(Codes,Y_F,Y_M)
        fig_code=None

    BaselineCodeScore_Mean,BaselineCodeScore_Std=GetScores(code_PrincipalMetricsArray)

    if LogToWandb:
        wandb.run.summary["%s_Diff_Mean"%MethodToUse]=BaselineCodeScore_Mean['Diff']
        wandb.run.summary["%s_Diff_Std"%MethodToUse]=BaselineCodeScore_Std['Diff']
        wandb.run.summary["%s_Ratio_Mean"%MethodToUse]=BaselineCodeScore_Mean['Ratio']
        wandb.run.summary["%s_Ratio_Std"%MethodToUse]=BaselineCodeScore_Std['Ratio']

    return fig_code,BaselineCodeScore_Mean,BaselineCodeScore_Std
