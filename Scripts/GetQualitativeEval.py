from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

## Figure 1

X, T, D, MHR, FHR, _ = data_loader.GetBatch(BatchSize=8, seg_len=hp.InputSegment)
[X, T, D] = [x.to('cpu') for x in [X, T, D]]
[X,T]=[input_norm_fun(x) for x in [X,T]]
x_code = enc(X)
[x,t,x_code]=[A.detach().cpu() for A in [X,T,x_code]]

fs=15
t_vec=np.linspace(0,4,len(x[0,0,:]))

fig1=plt.figure(figsize=(20,20))
plt.subplot(311)
yoffset=0
for ch_ind in range(5):
    yoffset = yoffset - min(x[0,ch_ind,:])
    plt.plot(t_vec,yoffset+x[0,ch_ind,:])
    yoffset = yoffset + max(x[0,ch_ind,:])
plt.xlim([0,2])
# plt.gca().get_xaxis().set_ticks([])
# plt.gca().get_yaxis().set_ticks([])
plt.xlabel('Time [sec]\n',fontsize=fs)
plt.ylabel('Abdominal\n channels',fontsize=fs)

plt.subplot(312)
yoffset=0
for ch_ind in range(3):
    yoffset = yoffset - min(t[0,ch_ind,:])
    plt.plot(t_vec,yoffset+t[0,ch_ind,:])
    yoffset = yoffset + max(t[0,ch_ind,:])
plt.xlim([0,2])
# plt.gca().get_xaxis().set_ticks([])
# plt.gca().get_yaxis().set_ticks([])
plt.xlabel('Time [sec]\n',fontsize=fs)
plt.ylabel('Thorax\n channels',fontsize=fs)



plt.subplot(313)
yoffset=0
for ch_ind in range(5):
    yoffset = yoffset - min(x_code[0,:,ch_ind])
    plt.plot(t_vec,yoffset+x_code[0,:,ch_ind])
    yoffset = yoffset + max(x_code[0,:,ch_ind])
plt.xlim([0,2])
# plt.gca().get_xaxis().set_ticks([])
# plt.gca().get_yaxis().set_ticks([])
plt.xlabel('Time [sec]\n',fontsize=fs)
plt.ylabel('Code',fontsize=fs)
# fig.show()
fig1.savefig(os.path.join(os.getcwd(),'SavedFigures','Sub%g_CahnnelsVsTime.png'%hp.Subjects[0]))
# fig.savefig(os.path.join(os.getcwd(),'SavedFigures','Sub%g_Test.png'%hp.Subjects[0]))



## Figure 2

X, T, H,Y_M,Y_F,Y_I = data_loader.GetTrajectory(BatchSize=2000,seg_len=hp.InputSegment)
[X,T,H]=[x.detach().cpu() for x in [X,T,H]]
[X,T]=[input_norm_fun(x) for x in [X,T]]
code_x=enc(X)
code_x=code_x.detach()
code_x=code_x.permute((0,2,1))
LowDim = 3
fs=15

X=X[:,1:,:]
T=T[:,1:,:]
X=X.reshape(X.shape[0],-1)
T=T.reshape(T.shape[0],-1)
H=H.reshape(H.shape[0],-1)
code_x=code_x.reshape(code_x.shape[0],-1)


[X,T,H,code_x]=[x**2 for x in [X,T,H,code_x]]
# [X,T,H,code_x]=[np.abs(x) for x in [X,T,H,code_x]]
UseTSNE = False

Y_T_M=torch.tensor([np.mod(x,int(Y_M[0])) for x in range(X.shape[0])])
Y_T_F=torch.tensor([np.mod(x,int(Y_F[0])) for x in range(X.shape[0])])
Names=['mECG','fECG']


fig2 = plt.figure(figsize=(13, 8))
gs = gridspec.GridSpec(2, 3)
for ic,c in enumerate([Y_T_M,Y_T_F]):
    ax = fig2.add_subplot(gs[ic, 0]) if LowDim == 2 else fig2.add_subplot(gs[ic, 0], projection='3d')
    ScatterHighDimensionalData(X, c, LowDim=LowDim, gax=ax,UseTSNE=UseTSNE)
    ax.set_axis_off()
    plt.set_cmap('twilight')
    if ic==0:
        ax.set_title('Abodiminal channels',fontsize=fs)
    ax.annotate('Colored by \n   %s' % Names[ic], xy=(0.5, 1), xytext=(-170, -80 - 8 * ic),
                xycoords='axes fraction', textcoords='offset points', size=fs)

    ax = fig2.add_subplot(gs[ic, 1]) if LowDim == 2 else fig2.add_subplot(gs[ic, 1], projection='3d')
    ScatterHighDimensionalData(T, c, LowDim=LowDim, gax=ax,UseTSNE=UseTSNE)
    ax.set_axis_off()
    plt.set_cmap('twilight')
    if ic==0:
        ax.set_title('Thorax channels',fontsize=fs)

    ax = fig2.add_subplot(gs[ic, 2]) if LowDim == 2 else fig2.add_subplot(gs[ic, 2], projection='3d')
    ScatterHighDimensionalData(code_x, c, LowDim=LowDim, gax=ax,UseTSNE=UseTSNE)
    ax.set_axis_off()
    plt.set_cmap('twilight')
    if ic==0:
        ax.set_title('Code',fontsize=fs)
#fig2.show()

fig2.savefig(os.path.join(os.getcwd(),'SavedFigures','Sub%g_ColoredEmbeddings.png'%hp.Subjects[0]))
# fig2.savefig(os.path.join(os.getcwd(),'SavedFigures','Sub%g_Test.png'%hp.Subjects[0]))


if hp.LogToWandb:
    wandb.log({"Qualitative Evaluations": [wandb.Image(fig1),
                             wandb.Image(fig2)]})