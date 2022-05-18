#TODOS:
# GetAE
##
import torch.nn as nn

## misc
def GetActivation(ActivationToUse):
    if ActivationToUse == 'Relu':
        return nn.ReLU()
        # self.Act = nn.Identity()
    elif ActivationToUse == 'Tanh':
        return nn.Tanh()
    elif ActivationToUse == 'None':
        return nn.Identity()
    else:
        print('Check activation')
        return None

## Normalization
def MinMaxPos(X):
    Span=(X.max(axis=2)[0]-X.min(axis=2)[0]).unsqueeze(2)
    tX=X/Span
    tX = (tX -0.5- tX.min(axis=2)[0].unsqueeze(2))
    return (1+tX)/2

def MinMax(X):
    Span=(X.max(axis=2)[0]-X.min(axis=2)[0]).unsqueeze(2)
    tX=X/Span
    tX = (tX -0.5- 0*tX.min(axis=2)[0].unsqueeze(2))
    return tX

def Zscore(X):
    tX = (X - X.mean(axis=2).unsqueeze(2))
    tX=tX/tX.norm(dim=2).unsqueeze(2)
    return tX

def EnhancePeaks(X):
    X = NormZscore(X)
    X = (X) ** (2)
    X = NormZscore(X)
    return X

def GetInputNormalizationFunction(NormInput):
    if NormInput == 'zscore':
        return Zscore
    elif ModelType == 'EP':
        return EnhancePeaks
    elif ModelType == 'MinMax':
        return MinMax
    else:
        print('Check input normalization function')
        return None


## TS DeepFC encoder
class TSDeepFCEncoder(nn.Module):
    def __init__(self, ActivationToUse='Relu', nChannels=1, Dim=5, UseSN=False):
        super().__init__()
        if UseSN:
            self.weight_norm = lambda x: x
        else:
            self.weight_norm = lambda x: nn.utils.spectral_norm(x)
        # self.Act=GetActivation(ActivationToUse)
        self.fc = nn.Linear(nChannels, Dim)  # for L=2000
        self.fc = nn.Sequential(
            nn.Linear(nChannels, 10 * Dim),
            nn.LeakyReLU(),
            nn.Linear(10 * Dim, 10 * Dim),
            nn.LeakyReLU(),
            nn.Linear(10 * Dim, Dim)
        )


    def forward(self, x):
        x = x.permute([0, 2, 1])
        x = self.fc(x)
        return x


class TSDeepFCDecoder2(nn.Module):
    def __init__(self, ActivationToUse='Relu', nChannels=1, nSamplesOut=2000, Dim=5, UseSN=False):
        super().__init__()
        if UseSN:
            self.weight_norm = lambda x: x
        else:
            self.weight_norm = lambda x: nn.utils.spectral_norm(x)
        self.nChannels = nChannels
        self.fc = nn.Linear(Dim * 2, nChannels)  # for L=2000
        self.fc = nn.Sequential(
            nn.Linear(Dim * 2, Dim * 2),
            nn.LeakyReLU(),
            nn.Linear(Dim * 2, Dim * 2),
            nn.LeakyReLU(),
            nn.Linear(Dim * 2, nChannels)
        )
    def forward(self, code_x, code_t):
        code = torch.cat((code_x, code_t), dim=2);
        out = self.fc(code)
        out = out.permute([0, 2, 1])
        return out


class TSDeepFCDecoder(nn.Module):
    def __init__(self, ActivationToUse='Relu', nChannels=1, nSamplesOut=2000, Dim=5, UseSN=False):
        super().__init__()
        if UseSN:
            self.weight_norm = lambda x: x
        else:
            self.weight_norm = lambda x: nn.utils.spectral_norm(x)
        self.nChannels = nChannels
        self.fc = nn.Linear(Dim + 3, nChannels)  # for L=2000
        self.fc = nn.Sequential(
            nn.Linear(Dim + 3, Dim + 3),
            nn.LeakyReLU(),
            nn.Linear(Dim + 3, Dim + 3),
            nn.LeakyReLU(),
            nn.Linear(Dim + 3, nChannels)
        )

    def forward(self, code_x, code_t):
        code = torch.cat((code_x, code_t), dim=2)
        out = self.fc(code)
        out = out.permute([0, 2, 1])
        return out


if 0:
    nChannels = 5
    nSamples = 2000
    x = torch.tensor(np.random.randn(32, nChannels, nSamples)).type(torch.FloatTensor)
    t = torch.tensor(np.random.randn(32, nChannels, nSamples)).type(torch.FloatTensor)
    enc = TSDeepFCEncoder(nChannels=nChannels)
    dec = TSDeepFCDecoder(nChannels=nChannels, nSamplesOut=nSamples)
    code_x = enc(x)
    code_t = enc(t)
    print(code_x.shape)
    print(dec(code_x, code_t).shape)


## TS conv

class MyTSConvEncoder(nn.Module):
    def __init__(self, ActivationToUse='Relu', nChannels=1, Dim=200, UseBN=True, UseSN=False):
        super().__init__()
        self.Act=GetActivation(ActivationToUse)
        if UseSN:
            self.weight_norm = lambda x: x
        else:
            self.weight_norm = lambda x: nn.utils.spectral_norm(x)
        self.Dim = Dim
        self.UseBN = UseBN
        self.UseSN = UseSN

        self.conv1 = nn.Conv1d(in_channels=nChannels, out_channels=8, kernel_size=3, padding=1)
        # self.maxpool2 = nn.MaxPool1d(2, return_indices=True)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
        if self.UseBN:
            self.bn4 = nn.BatchNorm1d(8)
        # self.maxpool5 = nn.MaxPool1d(2, return_indices=True)
        self.conv6 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        if self.UseBN:
            self.bn7 = nn.BatchNorm1d(8)
        # self.maxpool8 = nn.MaxPool1d(2, return_indices=True)
        self.conv9 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=11, padding=5)
        self.conv10 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=13, padding=6)
        # self.maxpool11 = nn.MaxPool1d(2, return_indices=True)
        self.conv12 = nn.Conv1d(in_channels=8, out_channels=Dim, kernel_size=3, padding=1)
        # self.conv13 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=7,padding=3)
        # self.maxpool14 = nn.MaxPool1d(2, return_indices=True)
        # self.fc=nn.Linear(250, Dim)  # for L=2000

        modules = [self.conv3, self.conv6, self.conv9, self.conv10, self.conv12]
        modules = [self.weight_norm(x) for x in modules]

    def forward(self, x):
        code = self.conv1(x)
        code = self.Act(code)  # print(code.shape)
        # code, self.ind2 = self.maxpool2(code)
        code = self.conv3(code)
        code = self.Act(code)  # print(code.shape)
        if self.UseBN:
            code = self.bn4(code)
        # code, self.ind5 = self.maxpool5(code)
        code = self.conv6(code)
        code = self.Act(code)  # print(code.shape)
        if self.UseBN:
            code = self.bn7(code)
        # code, self.ind8 = self.maxpool8(code)
        code = self.conv9(code)
        code = self.Act(code)  # print(code.shape)
        code = self.conv10(code)
        code = self.Act(code)  # print(code.shape)
        # code, self.ind11 = self.maxpool11(code)
        code = self.conv12(code)
        code = self.Act(code)  # print(code.shape)
        return code.permute([0, 2, 1])


class MyTSConvDecoder2(nn.Module):
    def __init__(self, ActivationToUse='Relu', nChannels=1, Dim=200, UseBN=True, UseSN=False):
        super().__init__()
        self.Act=GetActivation(ActivationToUse)
        if UseSN:
            self.weight_norm = lambda x: x
        else:
            self.weight_norm = lambda x: nn.utils.spectral_norm(x)
        self.Dim = Dim
        self.UseBN = UseBN
        self.UseSN = UseSN

        # self.conv1 = nn.Conv1d(in_channels=nChannels, out_channels=8, kernel_size=3,padding=1)
        # # self.maxpool2 = nn.MaxPool1d(2, return_indices=True)
        # self.conv3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5,padding=2)
        # if self.UseBN:
        #     self.bn4 = nn.BatchNorm1d(8)
        # # self.maxpool5 = nn.MaxPool1d(2, return_indices=True)
        # self.conv6 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3,padding=1)
        # if self.UseBN:
        #     self.bn7 = nn.BatchNorm1d(8)
        # # self.maxpool8 = nn.MaxPool1d(2, return_indices=True)
        # self.conv9 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=11,padding=5)
        # self.conv10 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=13,padding=6)
        # # self.maxpool11 = nn.MaxPool1d(2, return_indices=True)
        # self.conv12 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=3,padding=1)

        self.dconv1 = nn.ConvTranspose1d(in_channels=2 * Dim, out_channels=8, kernel_size=3, padding=1)
        self.dconv2 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=13, padding=6)
        self.dconv3 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.dconv4 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
        self.dconv5 = nn.ConvTranspose1d(in_channels=8, out_channels=nChannels, kernel_size=3, padding=1)
        modules = [self.dconv1, self.dconv2, self.dconv3, self.dconv4, self.dconv5]
        modules = [self.weight_norm(x) for x in modules]

    def forward(self, code_x, code_t):
        code = torch.cat((code_x.permute([0, 2, 1]), code_t.permute([0, 2, 1])), dim=1);  # print(code.shape)
        code = self.dconv1(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv2(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv3(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv4(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv5(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        return code


class MyTSConvDecoder(nn.Module):
    def __init__(self, ActivationToUse='Relu', nChannels=1, Dim=200, UseBN=True, UseSN=False):
        super().__init__()
        self.Act=GetActivation(ActivationToUse)
        if UseSN:
            self.weight_norm = lambda x: x
        else:
            self.weight_norm = lambda x: nn.utils.spectral_norm(x)
        self.Dim = Dim
        self.UseBN = UseBN
        self.UseSN = UseSN

        self.dconv1 = nn.ConvTranspose1d(in_channels=(3 + Dim), out_channels=8, kernel_size=3, padding=1)
        self.dconv2 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=13, padding=6)
        self.dconv3 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.dconv4 = nn.ConvTranspose1d(in_channels=8, out_channels=8, kernel_size=5, padding=2)
        self.dconv5 = nn.ConvTranspose1d(in_channels=8, out_channels=nChannels, kernel_size=3, padding=1)
        modules = [self.dconv1, self.dconv2, self.dconv3, self.dconv4, self.dconv5]
        modules = [self.weight_norm(x) for x in modules]

    def forward(self, code_x, code_t):
        code = torch.cat((code_x.permute([0, 2, 1]), code_t.permute([0, 2, 1])), dim=1);  # print(code.shape)
        code = self.dconv1(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv2(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv3(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv4(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        code = self.dconv5(code)
        code = self.Act(code)  # print(code.shape)
        # plt.figure(); plt.plot(code[1,0,:].detach().cpu()); plt.show()
        return code


##
def GetAE(ModelType, ActivationToUse='Relu', nChannels=1, Dim=100, UseBN=True, UseSN=False):
    if ModelType == 'Fc':
        enc = FcEncoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)
        dec = FcDecoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)
    elif ModelType == 'ModPaper':
        enc = ModPaperEncoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim)
        dec = ModPaperDecoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim)
    elif ModelType == 'ModPaper2':
        enc = ModPaperEncoder2(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim)
        dec = ModPaperDecoder2(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim)
    elif ModelType == 'MyConv':
        enc = MyConvEncoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseBN=UseBN, UseSN=UseSN)
        dec = MyConvDecoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)
    elif ModelType == 'MyConv2':
        enc = MyConvEncoder2(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseBN=UseBN, UseSN=UseSN)
        dec = MyConvDecoder2(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)

    elif ModelType == 'TSFc':
        enc = TSEncoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)
        dec = TSDecoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)
    elif ModelType == 'TSDeepFc2':
        enc = TSDeepFCEncoder(ActivationToUse=ActivationToUse, nChannels=hp.nChannels, Dim=hp.Dim, UseSN=hp.UseSN)
        dec = TSDeepFCDecoder2(ActivationToUse=ActivationToUse, nChannels=hp.nChannels, Dim=hp.Dim, UseSN=hp.UseSN)
    elif ModelType == 'TSConv2':
        enc = MyTSConvEncoder(ActivationToUse=ActivationToUse, nChannels=hp.nChannels, Dim=hp.Dim, UseBN=hp.UseBN,UseSN=hp.UseSN)
        dec = MyTSConvDecoder2(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseBN=UseBN,UseSN=UseSN)
    elif ModelType == 'TSDeepFc':
        enc = TSDeepFCEncoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)
        dec = TSDeepFCDecoder2(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseSN=UseSN)
    elif ModelType == 'TSConv':
        enc = MyTSConvEncoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseBN=UseBN, UseSN=UseSN)
        dec = MyTSConvDecoder(ActivationToUse=ActivationToUse, nChannels=nChannels, Dim=Dim, UseBN=UseBN, UseSN=UseSN)
    else:
        print('Check type')
        return None, None

    return enc, dec
