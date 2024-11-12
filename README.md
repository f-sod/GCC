## GCC code , predicition of endocrine activity using tox21 NR assay 

This repository contains the materials to generate the result from the poser presented during GCC, including Jupyter Notebooks, data files, and annotations.
Since the file are too heavy, the link to the folder is given and get retrieved from the [GOOGLE DRIVE](https://drive.google.com/drive/folders/1yZM5IiQcOTA_glRf9iDWoc4F_oceoGRP?usp=sharing)
## Repository Structure

### Data Directory
- **Data/Annotations/**: Contains various annotation files including:
  - `pubchem_annotation1.txt`: Latest version run on october 2024, we use the 'Broad identifier' and or 'smiles' to query Pubchem en retrieved the compound identifier CID. This identifier being "international", help in overlappind dataset in a more accurate way that the compound name or the smiles of a compounds
  - `tox21_10k_library.tsv`: TSV file containing chemical annotations of all tested compounds during the TOX21, downloaded [here]().
  - `Tox21_assay_list.xltx`: The dataset used for estrogen receptor activity prediction, downloaded from [EPA]().
  - `Tox21_assay_aggregated.csv`: Tox21 assay output CONCATENATED INTO ONE UNIQUE FILE, initial files too heavy so concatenated output given 


### Cell_Profiles Directory
- **Data/CellProfiles/**: Directory containing cell profile pikled files processed and cleaned, output from [notebook 1 developed for MIMB](https://github.com/volkamerlab/MIMB_cellpainting_tutorial/blob/main/Notebooks/Part1-Data_Processing.ipynb).
  - `output_notebook_1.pkl` : Output of _Part1-Data_Processing.ipynb_ 
        

### Output Directory
- **Data/Output/**: Directory where the output of the Jupyter Notebooks will be stored. This directory is initially non-existing and will be created and generated after running the 1st notebook, it will contain:
  - `Tox21_activity_BBC047.csv`: Output of 
  - `Tox21_annotation.csv` : Output of 
  - `Tox21_Endocrine_activity.csv`: Output of 

### Notebooks Directory
- **Notebooks/**: Contains Jupyter Notebooks for different parts of the tutorial:
  - `Endocrine_activity_prediction.ipynb`: Notebook for machine learning tasks.
  - `Tox21.ipynb`:  Notebook to prepare Tox21 data.


### Other Files

- **README.md**: This file, provides an overview of the repository structure and contents.
- **LICENSE**: License information 
- **environemment.yml** : File listing environment dependencies.