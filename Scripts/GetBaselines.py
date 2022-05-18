ScoresPerBaselines=[]
for MethodToUse in BaselinesToComare:
    curr_fig,BaselineCodeScore_Mean,BaselineCodeScore_Std=GetScoresForBaeline(MethodToUse,data_loader,hp.Subjects[0],load_trained=LoadSavedBaselineModels,LogToWandb=hp.LogToWandb)
    ScoresPerBaselines.append((BaselineCodeScore_Mean,BaselineCodeScore_Std))
    if hp.LogToWandb:
        figs_list.append(curr_fig)
