X, T, H,Y_M,Y_F,Y_I= data_loader.GetBatch(BatchSize=2000,seg_len=2000)
[X,T,H]=[x.detach().cpu() for x in [X,T,H]]
[X,T]=[input_norm_fun(x) for x in [X,T]]
code_x=enc(X)

if hp.LogToWandb:
    fig_ab,ab_WeightedMetricsArray,ab_PrincipalMetricsArray,ab_MaxMetricsArray = PlotAndGetRatios(X,Y_F,Y_M,'Input')
    fig_code,code_WeightedMetricsArray,code_PrincipalMetricsArray,code_MaxMetricsArray = PlotAndGetRatios(code_x.permute([0,2,1]),Y_F,Y_M,'Ours')
    figs_list=[fig_ab,fig_code]
else:
    ab_WeightedMetricsArray,ab_PrincipalMetricsArray,ab_MaxMetricsArray = GetMetricsForCode(X,Y_F,Y_M)
    code_WeightedMetricsArray,code_PrincipalMetricsArray,code_MaxMetricsArray = GetMetricsForCode(code_x.permute([0,2,1]),Y_F,Y_M)

InputScore_Mean,InputScore_Std=GetScores(ab_PrincipalMetricsArray)
CodeScore_Mean,CodeScore_Std=GetScores(code_PrincipalMetricsArray)

if hp.LogToWandb:
    wandb.run.summary["Input_Diff_Mean"]=InputScore_Mean['Diff']
    wandb.run.summary["Input_Diff_Std"]=InputScore_Std['Diff']
    wandb.run.summary["Input_Ratio_Mean"]=InputScore_Mean['Ratio']
    wandb.run.summary["Input_Ratio_Std"]=InputScore_Std['Ratio']

    wandb.run.summary["Ours_Diff_Mean"]=CodeScore_Mean['Diff']
    wandb.run.summary["Ours_Diff_Std"]=CodeScore_Std['Diff']
    wandb.run.summary["Ours_Ratio_Mean"]=CodeScore_Mean['Ratio']
    wandb.run.summary["Ours_Ratio_Std"]=CodeScore_Std['Ratio']
