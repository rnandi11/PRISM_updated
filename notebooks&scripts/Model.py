import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,accuracy_score,make_scorer,roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from scipy.stats import randint
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from boruta import BorutaPy

import pickle

import warnings
warnings.filterwarnings('ignore')


# -------------------------import data-------------------------------- 
df=pd.read_csv('UKBB_preprocessed.csv')

# ---------------Create a column for presence of giant platelets-----------
df['giant_plt']=np.where(df['PDW']>16.5,1,0)



# -----------------Create a classification label----------------------

df_ar=df.to_numpy()
rows,columns=df_ar.shape
print('size of full dataset:',rows,columns)

# Assign condition for CH 

CH0=np.where(df['CH_score']<=4)
CH1=np.where((df['CH_score']>4) & (df['MN']==0))
CH2=np.where((df['CH_score']>4) & (df['MN']==1))

CH_ar=np.zeros((rows,))
CH_ar[CH0]=0
CH_ar[CH1]=1
CH_ar[CH2]=2

df['CH']=CH_ar

print("Number of participants in each class:",df['CH'].value_counts())



# -----------Split the data into train and test set--------------------
seed=345

data=df.copy()

X=data.drop('CH',axis=1)
y=data['CH']

X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

training_df=pd.concat([X_train,y_train], axis=1)
test_df=pd.concat([X_test,y_test], axis=1)

print('Training data size:', X_train.shape)
print('Test data size:', X_test.shape)

training_df.to_csv('training_df.csv')
test_df.to_csv('test_df.csv')


# --------------------Drop the columns not used as model features------------------


columns_to_drop=['Broad_ID','gene1','gene2','gene3','VAF1','VAF2','VAF3','CH_score','eid','WBC','haematocrit',
                 'hiscatret','year_of_assessment','year_HM','HM_type','CHRS','MN']

X_train_mod= X_train.drop(columns=columns_to_drop) 
X_test_mod=X_test.drop(columns=columns_to_drop) 
X_test_mod.head()


# ---------Run a Balanced Random Forest Model----------------------------

# Define the hyperparameters to search for RandomForest
param_dist = {
    'n_estimators': randint(10, 300),       # Number of trees in the forest
    'max_depth': randint(1, 30),            # Maximum depth of the tree
    'min_samples_split': randint(2, 20),    # Minimum samples required to split a node
    'min_samples_leaf': randint(1, 20),     # Minimum samples required at a leaf node
    'max_features': ['log2'],               # Features to consider when looking for the best split
    'bootstrap': [True],                    # Whether bootstrap samples are used when building trees
    'criterion': ['gini'],                  # Function to measure the quality of a split
}

# Define a custom scoring function for multi-class AUC
roc_auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)


# Initialize the BalancedRandomForest model
model = BalancedRandomForestClassifier(replacement=True, sampling_strategy='all', random_state=seed)


# Randomized Search over hyperparameters
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=500,  # Number of trials
    scoring=roc_auc_scorer,  # Use roc_auc score for evaluation
    n_jobs=-1,  # Use all available cores
    cv=5,  # 5-fold cross-validation
    random_state=seed
)

# Fit the model on the training data
random_search.fit(X_train_mod, y_train)

# Best model after RandomizedSearchCV
best_model_RF = random_search.best_estimator_

# Print the best model parameters
print("Best Model Parameters:")
print(random_search.best_params_)

# Predictions on the test set
y_pred = best_model_RF.predict(X_test_mod)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))



# ------------------------ Feature selection using Boruta -------------------------------------

# Initialize RandomForest model
model=best_model_RF

# Initialize Boruta
boruta_selector = BorutaPy(estimator=model, verbose=2, max_iter= 100, random_state=seed)

# Fit Boruta
boruta_selector.fit(X_train_mod, y_train)

# Get selected features
selected_features = X_train_mod.columns[boruta_selector.support_].tolist()
print("Selected Features by Boruta:")
print(selected_features)

# Optionally, get ranking of all features
feature_ranks = list(zip(X_train_mod.columns, boruta_selector.ranking_))
feature_ranks.sort(key=lambda x: x[1])  # Sort by importance
print("Feature Rankings:")
for feature, rank in feature_ranks:
    print(f"{feature}: Rank {rank}")


# ------------------- Keep only selected features ---------------------------------------

cols_to_drop=X_test_mod.columns.difference(selected_features)


X_train_bor = X_train_mod.drop(columns=cols_to_drop)
X_test_bor = X_test_mod.drop(columns=cols_to_drop)
y_train_bor = y_train.copy()
y_test_bor = y_test.copy()

training_df_bor=pd.concat([X_train_bor,y_train_bor], axis=1)
test_df_bor=pd.concat([X_test_bor,y_test_bor], axis=1)


# ------------------------------- Retune model with selected features -----------------------

# Fit the model on the training data
random_search.fit(X_train_bor, y_train_bor)

# Best model after RandomizedSearchCV
best_model_boruta = random_search.best_estimator_

# Print the best model parameters
print("Best Model Parameters:")
print(random_search.best_params_)

# Predictions on the test set
y_pred = best_model_boruta.predict(X_test_bor)

# Accuracy Score
accuracy = accuracy_score(y_test_bor, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# Print the classification report
print("Classification Report:")
print(classification_report(y_test_bor, y_pred))


# ---------------------------------- AUC, threshold calculatio and plot ------------------------------


colors = ['blue', 'red', 'green', 'purple']  # Adjust for more classes


# Binarize the labels for multi-class classification
n_classes = len(np.unique(y_test_bor))  # Number of classes
y_test_bin = label_binarize(y_test_bor, classes=np.unique(y_test_bor))  # Convert to one-hot encoding
y_pred_proba = best_model_boruta.predict_proba(X_test_bor)  # Get predicted probabilities for each class

# Plot ROC curve
plt.figure(figsize=(4, 4))
plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.5)  # Diagonal reference line

for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])  # Compute ROC for class i
    roc_auc = auc(fpr, tpr)  # Compute AUC
    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    ix=np.argmax(tpr-fpr)
    print("class, theshold, gmeans:",i, thresholds[ix])


# Labels and legend
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Multiclass ROC (One vs Rest")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------

# Optimal thresholds (as computed earlier)
# Order: class 0, class 1, ..., class 5
thresholds = np.array([0.3857391006797291,0.38776188909857673,0.2380357493945122])

# Get predicted probabilities from the best model
y_proba = best_model_boruta.predict_proba(X_test_bor)  # shape: (n_samples, 6)

# Initialize an empty list for final predictions
y_pred_custom = []

# Iterate over each sample's probability vector
for prob_vector in y_proba:
    # Identify classes where probability exceeds its threshold
    passing_classes = np.where(prob_vector >= thresholds)[0]
    
    if passing_classes.size > 0:
        # If one or more classes pass, choose the one with highest probability among them
        chosen_class = passing_classes[np.argmax(prob_vector[passing_classes])]
    else:
        # Otherwise, default to the class with the maximum probability
        chosen_class = np.argmax(prob_vector)
    
    y_pred_custom.append(chosen_class)

# Convert list to numpy array
y_pred_custom = np.array(y_pred_custom)

# Evaluate the predictions
accuracy = accuracy_score(y_test_bor, y_pred_custom)
print(f"\nAccuracy Score: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test_bor, y_pred_custom))

# Compute confusion matrix
cm = confusion_matrix(y_test_bor, y_pred_custom)

# Plot confusion matrix with a custom colormap
cmap = sns.color_palette("Blues", as_cmap=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(4, 3))
ax = plt.gca()
disp.plot(cmap=cmap, ax=ax, values_format='d', text_kw={"fontsize": 14})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()



# ----------------------------- Feature Importance With SHAP ------------------------------------

import shap

loaded_model.fit(X_train_bor,y_train_bor)
# Fits the explainer
explainer = shap.Explainer(loaded_model.predict, X_test_bor)

# Create SHAP explainer for the model
explainer = shap.Explainer(loaded_model.predict_proba, X_test_bor, check_additivity=False)

# Compute SHAP values
shap_values = explainer(X_test_bor)  # Returns an array of SHAP values (one per class)


# Select the class to visualize (e.g., class 0)
class_index = 1  # Change this to visualize a different class
shap_values_class = shap_values[:, :, class_index]  # Extract SHAP values for the chosen class

# Set figure size
plt.rcParams["figure.figsize"] = [6, 4]

# Plot SHAP summary plot for the chosen class
shap.summary_plot(shap_values_class, X_test_bor)



# ----------------------- Save Model and features ------------------------------------

import pickle

# Bundle model and feature list
model_package = {
    'model': best_model_boruta,
    'features': feature_list  # this should be a list of column names or indices
}

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("Model package saved successfully as model.pkl")










