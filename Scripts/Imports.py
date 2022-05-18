





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch import autograd
from torch.autograd import Variable


import os
import time
from tqdm import tqdm
import wandb
import pickle
from tabulate import tabulate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib
# import tkinter
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')

import scipy as sc
# import scipy.linalg
import statsmodels.api as sm
from scipy.signal import medfilt
from scipy import signal
# from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq, fftshift,ifft
import echotorch
import echotorch.nn as etnn
import padasip as pa
import numpy as np



from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split


np.random.seed(42)
torch.manual_seed(42)