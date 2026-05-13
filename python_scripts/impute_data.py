import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pickle
import os


# Load the unscreened dataset, input the actual file path
df_unscreened=pd.read_csv("path/to/unscreened_data.csv")


#-------------------------------------------------------------------------------------------------------------------
# Data Imputation
def impute(data, columns_to_impute):

	# Scale the dataframe 
	scaler = StandardScaler()
	data_scaled = pd.DataFrame(
    scaler.fit_transform(data[columns_to_impute]),columns=columns_to_impute,index=data.index)

    # KNN imputer now sees all columns when computing neighbor distances
    imputer = KNNImputer(n_neighbors=5)
    imputed_scaled = imputer.fit_transform(data_scaled)

    # Convert imputed scaled data back to a DataFrame
    data_imputed_scaled = pd.DataFrame(imputed_scaled, columns=columns_to_impute, index=data.index)

    # Inverse transform the full dataframe to get back original scale
    data_imputed = pd.DataFrame(scaler.inverse_transform(data_imputed_scaled),columns=columns_to_impute,index=data.index)

    # Only write back the columns we intended to impute
    data[columns_to_impute] = data_imputed[columns_to_impute]

    return data

# -------------------------------------------------------
def main():

	# Run Imputer
	columns_to_impute=['RBC','Hbconc', 'MCV', 'RDW', 'platelet','plateletcrit','lymphocyte','monocyte', 'neutrophil',
	'eosinophil', 'reticulocyte','age','giant_plt']

	df_imputed= impute(df_unscreened,columns_to_impute)
	df_imputed['giant_plt'] = np.where(df_imputed['giant_plt'] >= 0.5, 1, 0)

	df_imputed.to_csv('df_unscreened_imputed.csv')
	

if __name__ == "__main__":
	main()  
