class HP():
    def __init__(self,):
        # pre-processing and time segmentations
        self.InputSegment = 2000  # segment window length in samples
        self.nChannelsAb = 24
        self.nChannelsCh = 3
        self.Subjects = [46]  # list of subjects to process
        self.window_len = 101  # window length for median filter
        self.band_width = 80  # BW for LP filter (cutoff)
        self.UseSynthetic = -1  # for using synthetic dataset

        # architecture
        self.ModelType = 'TSConv'
        self.ModelDim = 5
        self.SameCondEncoder = False
        self.Disc_UseAct = True
        self.ActivationToUse = 'Tanh'
        self.NormInput = 'zscore'
        self.UseBN = True #batch norm
        self.UseSN = True #spectral norm
        self.CaeFac = 0 #factor of the contractive ae loss
        self.R1Fac = 0 #factor of the R1 loss

        # training
        self.NumberOfBatches = 50_000
        self.AeFac=1
        self.BatchSize = 8
        self.TrajectoriesAsInputs = False #sequential segments if True, othewise randomly drawn segements
        self.LearningRateAE = 1e-4
        self.LearningRateD = 1e-4
        self.UseScheduler = False
        self.WeightDecay = 0
        self.min_lr = 1e-5
        self.decay_step_size = 10000
        self.lr_decay_factor = 0.1
        self.milestones = [self.decay_step_size]
        while self.milestones[-1] < self.NumberOfBatches:
            self.milestones.append(self.milestones[-1] + self.decay_step_size)

        # losses
        self.Beta = 0.01
        self.AeFac = 5
        self.AeLoss = 'L2'
        self.IndLoss = 'ScaleInvMSE'
        self.Margin = 0.5 #for contrastive loss

        # misc + logging
        self.GPU = 0
        self.LogToWandb=False
        self.WadbUsername='XXX'
        self.ProjectName='ICA'
        self.ExpName='DefaultExperimentName'


class BaselinesHP():
    def __init__(self,):

        self.InputSegment = 2000
        self.nChannelsAb = 24
        self.MethodToUse = 'OLS'

        self.nTaps=5 #tapped delay line (relevant for all baselines)

        #RLS,LMS
        self.mu = 0.1

        #ESN
        self.ridge_param = 0.0001
        self.hidden_dim = 50
        self.leaky_rate = 0.4
        self.spectral_radius=0.4

        #ADALINE
        self.eta = 1
        self.n_iter=15


        #misc+logging
        self.Subjects=[46]
        self.LogToWandb=True
        self.peaks_range = 25
        self.Navg = 1000

def GetModifiedConf(base_conf,keys,vals):
    tmp=base_conf.copy()
    for (key,val) in zip(keys,vals):
        tmp[key]=val
    return tmp





