
import torch.nn as nn
##
def AbsScaleLossModule(x, y):
    zsx = x - x.mean(axis=1).unsqueeze(1)
    zsx = zsx / zsx.norm(dim=1).unsqueeze(1)

    zsy = y - y.mean(axis=1).unsqueeze(1)
    zsy = zsy / zsy.norm(dim=1).unsqueeze(1)

    return nn.MSELoss()(torch.abs(zsx), torch.abs(zsy))

class PearsonDisc(nn.Module, ):
    def __init__(self, z_dim=200 * 3):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 20),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(100, 100),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(100, 100),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(100, 50),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(50, 20),
            # nn.LeakyReLU(0.2, True),
            nn.Linear(20, 1),
        )

    def forward(self, e):
        return self.net(e).squeeze(2)

def PearsonLossModule(x, y):
    x = X_disc(x)
    y = T_disc(y)

    zsx_ub = x - x.mean(axis=1).unsqueeze(1)
    zsx = zsx_ub / zsx_ub.norm(dim=1).unsqueeze(1)
    #
    zsy_ub = y - y.mean(axis=1).unsqueeze(1)
    zsy = zsy_ub / zsy_ub.norm(dim=1).unsqueeze(1)

    pearson_corr = 1 - torch.mean(torch.sum(zsy * zsx, axis=1) ** 2)
    return pearson_corr

##
def GetReconstructionLossFun(AeLoss):
    if AeLoss=='L2':
        return lambda x, y: 1e4*nn.MSELoss()(x,y)
    elif AeLoss=='L1':
        # return  nn.L1Loss()
        return lambda x_dec, X: (torch.sum(torch.abs(X - x_dec))) / len(X.flatten())  # equiv to nn.L1Loss()(x_dec,X)
    elif AeLoss == 'EnhancedPeaks':
        return lambda x, y: 100 * nn.MSELoss()(x ** 2, y ** 2) + nn.MSELoss()(x, y)
    elif AeLoss=='NormalizedL2':
        return lambda x_dec, X: (torch.norm(X - x_dec) ** 2) / (torch.norm(X) ** 2)  # /len(X.flatten())
    elif AeLoss.startswith('EP'):
        weight = int(hp.AeLoss[2:])
        return lambda x_dec, X: ((torch.norm(X - x_dec) ** 2) / (torch.norm(X) ** 2) + weight * (
                torch.norm(X ** 2 - x_dec ** 2) ** 2) / (torch.norm(X ** 2) ** 2))  # /len(X.flatten()
    else:
        print('Check reconstruction loss function')

def GetIndependenceLossFun(IndLoss):
    if IndLoss=='L2':
        return nn.MSELoss()
    elif IndLoss=='L1':
        return nn.L1Loss()
    elif IndLoss == 'ScaleInvMSE':
        return lambda x, y: 1e4 * AbsScaleLossModule(x, y)
    elif IndLoss=='Pearson':
        return lambda x, y:  PearsonLossModule(x, y)
    elif IndLoss=='Contrastive':
        return ContrastiveLoss
    elif IndLoss=='BCE':
        return BCELoss
    elif IndLoss=='MyBCE':
        return MyBCELoss
    elif IndLoss=='FacBCE':
        return FactorBCELoss
    elif (IndLoss == 'LS' or IndLoss == 'NeuralLS'):
        return LSLoss
    elif IndLoss == 'CCA':
        # cca_loss = Cca_loss(device=device, outdim_size=62)
        return CCALoss
    elif IndLoss == 'TSCCA':
        # cca_loss = TS_Cca_loss(device=device)
        return TSCCALoss
    elif IndLoss == 'TSCorel':
        return TS_Corel_Loss
    elif IndLoss=='SlimPearson2':
        return lambda x, y: 1e4 * PearsonLoss(x, y)
    elif IndLoss=='MyPearson':
        return lambda x, y:  MyPearsonLoss(x, y)
    else:
        print('Check independence loss function')
