# Singapore Grab AI Challenge Submission (Safety)

Model architecture & workflow explanations are summarized in short presentation file: Grab AI Challenge Presentation - Safety - Lewis.pdf

Computing infrastructure requirements: Anaconda Python Installation, 40GB storage space, 32GB ram

Compute Infrastructure Preparation Instructions:
  1) Download this entire repository
  2) Place input feature file and label csv file into the same folder
  3) Download trained models and essential utility files from: https://drive.google.com/open?id=1F2ylS0u4JVIfD3e1lwGnSsMOnMZtID5z
  4) Unzip folder and place it in same directory as scripts
  5) If required install Anaconda, use conda env if you wish to
  6) pip install tensorflow
  7) pip install lightgbm
  8) Install dependencies if the module cannot be found during running of scripts: numpy, scikit-learn, h5py

Computing Scripts Instructions:
  1) Open script 1 and replace input features file and label file with the test file names
  2) Open script 4 and put in number of cores of the machine the script is going to run on
  3) Replace the shebang of each script with your own anaconda interpreter path
  4) Run script 1
  5) Run script 2
  6) Run script 3 (nohup recommended) - this takes time, you can read the pdf presentation to understand the pipeline architecture more
  7) pip install tsfresh - this is due to tsfresh requiring an older pandas version to work
  8) Run script 4 (nohup recommended)
  9) Run script 5 - predictions are exported into a csv file, if encounter error here, pip install pandas
  10) ROC AUC plot - exported as ROC_AUC_Plot.jpg
  
