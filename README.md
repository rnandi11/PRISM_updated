<h2>PRISM - A machine learning model using routine blood counts to prioritize sequencing for high-risk clonal hematopoiesis</h2>
This study used routine blood counts and demographic data from UK Biobank to train a Balanced Random Forest model to predict high-risk Clonal Hematopoiesis. 


<h3> Python Scipts </h3>

1. `model_concise.py` - This python script can be independently run to generate the baseline model from the preprocessed dataset.
2. `final_model.py` - This python script can be run to generate the final model from the baseline model (after threshold selection, and refitting on final feature set) on which performance is tested.
3. `impute_data.py` - Script for implementing KNN imputation for missing data.
4. `performance_evaluation.py` - This script generates figures and files of model performance metrics and CHRS benchmarking, and can be used on external validation sets.
