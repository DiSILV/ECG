## Initilizations

import os
exec(open(os.path.join("Scripts", "Imports.py")).read())
from Utilities.Data_Utils import *
from Utilities.Hyperparameters_Utils import *
from Utilities.Model_Utils import *
from Utilities.Loss_Utils import *
from Utilities.Baselines_Utils import *
from Utilities.Evaluation_Utils import *
ModelsFolder='SavedModels';FiguresFolder='SavedFigures'
np.random.seed(42)
torch.manual_seed(42)

## Configurations

GPU=-1 #set -1 for cpu
LoadSavedModels=True
LoadSavedBaselineModels=True

#in order to reproduce figures + tables
SubjectsList=np.arange(1,61)
BaselinesToComare=["ADALINE","ESN","RLS","LMS"]

#in order to reproduce figures only
# SubjectsList=[46]
# BaselinesToComare=[]

LogToWandb = False
WadbUsername = 'YourUsername'

## Experiments

ConfigurationsToExecute=[GetModifiedConf({},['Subjects','GPU'],[[x],GPU]) for x in SubjectsList]
DiscardedSubjects=[]
ResultsMeanDict,ResultsStdDict={'Ratio':[],'Diff':[]},{'Ratio':[],'Diff':[]}
for ie,e in tqdm(enumerate(ConfigurationsToExecute)):
    1
    try:
        #initializations
        hp = HP()
        [exec('hp.%s=%s' % (f, 'e[\'%s\']' % f)) for f in e.keys()]
        hp.ProjectName = 'ResultsSummary_Test'
        hp.LogToWandb = LogToWandb
        hp.WadbUsername = WadbUsername
        hp.ExpName = '_'.join(['%s=%s' % (f, hp.__getattribute__(f)) for f in e.keys() if not (f == 'GPU')])
        device = torch.device("cuda:%g" % hp.GPU if (torch.cuda.is_available() and hp.GPU >= 0) else "cpu")
        if hp.LogToWandb:
            wandb.login()
            wandb.init(project=hp.ProjectName, entity=hp.WadbUsername)
            wandb.run.name = hp.ExpName
            config=wandb.config
            config.args = vars(hp)

        #load data
        data_loader = MyDataLoader(Subjects=hp.Subjects, nChannelsAb=hp.nChannelsAb, nChannelsCh=hp.nChannelsCh, window_len=hp.window_len, band_width=hp.band_width, device=device)
        if data_loader.SubjectsDict[hp.Subjects[0]]['sigLen']<5000:
            #discard short recordings
            DiscardedSubjects.append('Experiment #%g, subject %g'%(ie,hp.Subjects[0]))
            continue

        #get model
        exec(open(os.path.join("Scripts", "GetModel.py")).read())

        #qualitative evaluations
        exec(open(os.path.join("Scripts", "GetQualitativeEval.py")).read())

        #objective evaluations
        exec(open(os.path.join("Scripts", "GetObjectiveEval.py")).read())

        #compare to baselines
        exec(open(os.path.join("Scripts", "GetBaselines.py")).read())

        #summarize results in tables
        for metric in ['Ratio','Diff']:
            ResultsMeanDict[metric].append([hp.Subjects[0],InputScore_Mean[metric],CodeScore_Mean[metric]]+[x[0][metric] for x in ScoresPerBaselines])
            ResultsStdDict[metric].append([hp.Subjects[0],InputScore_Std[metric],CodeScore_Std[metric]]+[x[1][metric] for x in ScoresPerBaselines])

        if hp.LogToWandb:
            wandb.log({"Objective Evaluations": [wandb.Image(x) for x in figs_list]})
        plt.close('all')

    except Exception as inst:
        DiscardedSubjects.append('Experiment #%g, subject %g'%(ie,e['Subjects'][0]))
        print(type(inst))
        print(inst.args)
        print(inst)
        
for metric in ['Ratio','Diff']:
    ResultsMeanDict[metric]=np.array(ResultsMeanDict[metric])
    ResultsStdDict[metric]=np.array(ResultsStdDict[metric])

## Save and print results
exec(open(os.path.join("Scripts", "PrintTables.py")).read())
np.save(os.path.join('SavedTables','Results_Mean'),ResultsMeanDict)
np.save(os.path.join('SavedTables','Results_Std'),ResultsStdDict)
with open(os.path.join('SavedTables','Results_PerSubject.txt'), 'w') as f:
    f.write(tab_per_subject)
with open(os.path.join('SavedTables','Results_Aggregated.txt'), 'w') as f:
    f.write(tab_agg)