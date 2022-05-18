This code reproduces the experimental results from the section "Fetal ECG extraction".

# Installation
1. Clone the repository to your local directory.
2. Install the requirements: `pip install -r requirements.txt`.

# Folders
This repository includes the following folders:
- `Data` - This folder contains the recording for each subject. This data was downloaded from: https://www.physionet.org/content/ninfea/1.0.0./ and pre-processed using the pre-processing scripts supplied by the authors.
For each subject X there exists two files:
    - "X_E.mat" - contains the ECG recordings of subject X.
    - "X_D.mat" - contains the Doppler recordings of subject X.
- `SavedModels` - This folder contains trained pytorch models. Using these trained models is optional.
- `SavedBaselines` - This folder contains trained baseline models. Using these trained models is optional.
- `SavedTables` - After executing the code, the tabular results (Table 2 and Table 6 in the paper) will be written to this folder in a Latex format.
- `SavedFigures` - After executing the code, the figures (Figure 5 and Figure 6 in the paper) will be written to this folder in a png format.

# Execution 
The script `Main.py` reproduces the results from the paper, it is also available in as a jupyter notebook: `Main.ipynb`.
The outputs of the script will be written to  the folders `SavedTables` and `SavedFigures`.

# Changing the configurations
The configurations are set in the "Configurations" section (lines 25-35 in `Main.ipynb` and lines 17-30 in `Main.py`).
- `GPU` - specifies whether  to use a GPU or not.
- `LoadSavedModels` - specifies whether to use trained models, or reproducing the training process as well (when using GPU it takes ~20min for each subject).
- `LoadSavedBaselines` - specifies whether to use trained baseline models.
- `SubjectsList` - a list of subjects to examine (the qualitative results in the paper are reported for subject #46).
- `BaselinesToComare` - a list of baselines to compare to.
- `LogToWandb` - specifies whether to log the results to W&B - https://wandb.ai  (requires an account). 
   If set to `True`, then it is required to modify  `WadbUsername` accordingly. 
   Logging to W&B is recommended, further qualitative results that were discarded from the paper are presented when using W&B.  
  
# Changing the hyperparameters
The hyperparameters are listed in `Utilities\Hyperparameters_Utils.py`. The class `HP` includes the hyperparameters of our model. 
The class `BaselinesHP` includes the hyperparameters of the considered baselines.
  
# Credits
The code for the baseline models was implemented  using the following repositories:
- LMS, RLS - https://matousc89.github.io/padasip/
- ESN  - https://github.com/nschaetti/EchoTorch/
- ADALINE -  https://github.com/Natsu6767/Adaline
