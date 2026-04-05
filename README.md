<h2>PRISM - A machine learning model using routine blood counts to prioritize sequencing for high-risk clonal hematopoiesis</h2>
This study used routine blood counts and demographic data from UK Biobank to train a Balanced Random Forest model to predict high-risk Clonal Hematopoiesis. 

<h4> Notebooks </h4>

1. `Data_Preprocess.ipynb` - Running this script will create the preprocessed dataset from UK Biobank used for model training and internal validation.
Requires access to UK Biobank population data, WES data and the linker file. Steps involve extracting relevant features from datasets,
renaming variables, calculating age, excluding individuals with previous hematological malignancies, assigning CH status, calculating molecular CH scores and 
CH risk scores (CHRS), and removing rows with missing blood counts.
2. `UKBB_EDA.ipynb` - This script performs exploratory data analysis on the preprocessed dataset.
3. `model_derivation.ipynb` - Defines output classes, splits the data into training and test sets, compares various classifiers, trains a balanced random-forestclassifier
   on the training set to derive the baseline model, performs feature importance tests using SHAP and Boruta.
4. `model_thresholds.ipynb` - Retrains model on subset of original features, and adjusts decision thresholds to maximize recall. This script creates the
   final model `3class_BRF_giantplt_ASH_latest.pkl` used for all analysis in the manuscript.
5. `Model_Evaluation_Analysis.ipynb` - Contains summary stats of training, test and external validation sets. Reports model performance, and benchmarks against the
   CHRS risk stratification for both the internal test and external validation set.
6. `CumulativeIncidences.ipynb` - Contains Kaplan-Meier survival plots, and Cox Proportional Harard Ratio analysis for incidence of myeloid malignancies
   and cardiovascular diseases for the PRISM predicted classes in the UKB test set.

<h4> Python Scipts </h4>

1. `model_concise.py` - This python script can be independently to generate the final model from the preprocessed dataset.
2. `impute_data.py` - Script for implementing KNN imputation for missing data.
3. `model_evaluation.py` - This script generates figures and files of model performance metrics and CHRS benchmarking, and can be used on external validation sets.
