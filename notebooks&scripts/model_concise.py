import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, ConfusionMatrixDisplay, 
                             roc_auc_score, make_scorer, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from imblearn.ensemble import BalancedRandomForestClassifier
from boruta import BorutaPy
import shap
import joblib
import os
import pickle


seed=345

def split_data(data,seed=seed):
    X=data.drop('CH',axis=1)
    y=data['CH']

    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    training_df=pd.concat([X_train,y_train], axis=1)
    test_df=pd.concat([X_test,y_test], axis=1)

    print('Training data size:', X_train.shape)
    print('Test data size:', X_test.shape)

    training_df.to_csv('training_df.csv')
    test_df.to_csv('test_df.csv')

    return X_train,X_test,y_train,y_test


def train_balanced_rf(X_train, y_train, seed=seed, n_iter=100):
    """Train Balanced Random Forest with Randomized Search"""
    param_dist = {
        'n_estimators': randint(10, 300),
        'max_depth': randint(1, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['log2'],
        'bootstrap': [True],
        'criterion': ['gini'],
    }

    roc_auc_scorer = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)

    model = BalancedRandomForestClassifier(replacement=True, sampling_strategy='all', random_state=seed)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=roc_auc_scorer,
        n_jobs=-1,
        cv=5,
        random_state=seed
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def run_boruta(model, X_train, y_train, seed=seed):
    """Run Boruta feature selection"""
    boruta_selector = BorutaPy(estimator=model, verbose=2, max_iter=100, random_state=seed)
    boruta_selector.fit(X_train.values, y_train)

    selected_features = X_train.columns[boruta_selector.support_].tolist()
    feature_ranks = list(zip(X_train.columns, boruta_selector.ranking_))
    feature_ranks.sort(key=lambda x: x[1])

    return selected_features, feature_ranks


def evaluate_model(model, X_test, y_test, thresholds=None, output_file=None):
    """Evaluate model predictions using standard and thresholded probability-based predictions"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if output_file:
        with open(output_file, 'w') as f:
            f.write("Standard Prediction Report\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(report)
    
    # Plot confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title("Standard Confusion Matrix")
    plt.show()

    # ROC and threshold optimization
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    colors = ['blue', 'red', 'green']

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    if thresholds is None:
        thresholds = []
        for i in range(n_classes):
            fpr, tpr, thr = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ix = np.argmax(tpr - fpr)
            thresholds.append(thr[ix])
            plt.plot(fpr, tpr, color=colors[i], label=f'Class {i} (AUC={roc_auc:.2f})')
            if output_file:
                with open(output_file, 'a') as f:
                    f.write(f"\nClass {i}: optimal threshold = {thr[ix]:.4f}, AUC = {roc_auc:.4f}\n")
        thresholds = np.array(thresholds)

    plt.legend()
    plt.title("Multiclass ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)
    plt.show()

    # Threshold-based prediction
    y_pred_custom = []
    for probs in y_pred_proba:
        passing_classes = np.where(probs >= thresholds)[0]
        if passing_classes.size > 0:
            chosen = passing_classes[np.argmax(probs[passing_classes])]
        else:
            chosen = np.argmax(probs)
        y_pred_custom.append(chosen)

    y_pred_custom = np.array(y_pred_custom)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    report_custom = classification_report(y_test, y_pred_custom)
    cm_custom = confusion_matrix(y_test, y_pred_custom)

    if output_file:
        with open(output_file, 'a') as f:
            f.write("\nCustom Threshold Prediction Report\n")
            f.write(f"Accuracy: {acc_custom:.4f}\n")
            f.write(report_custom)

    ConfusionMatrixDisplay(confusion_matrix=cm_custom).plot(cmap='Blues')
    plt.title("Custom Threshold Confusion Matrix")
    plt.show()

    return thresholds


def run_shap(model, X_test, y_test):
    """Compute and display SHAP feature importance for a given class"""
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    explainer = shap.Explainer(model.predict_proba, X_test, check_additivity=False)
    shap_values = explainer(X_test)
    for i in range(n_classes):
        shap.summary_plot(shap_values[:, :, i], X_test, show=True, title=f"Class {i}")



def run_pipeline(data, seed=seed, output_path="model_results.txt"):
    """Main pipeline to train, select features, evaluate and explain the model"""

    # Step 1: Split into training and test set
    X_train,X_test,y_train,y_test=split_data(data)

    # Step 1: Train initial model
    print("Training initial Balanced Random Forest...")
    best_model, best_params = train_balanced_rf(X_train, y_train, seed=seed)
    print("Best initial parameters:", best_params)

    # Step 2: Feature Selection
    print("Running Boruta feature selection...")
    selected_features, ranks = run_boruta(best_model, X_train, y_train, seed=seed)
    print("Selected Features:", selected_features)

    # Step 3: Filter dataset
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Step 4: Retrain with selected features
    print("Retraining with selected features...")
    best_model_boruta, best_params_boruta = train_balanced_rf(X_train_sel, y_train, seed=seed)
    print("Best Boruta-tuned parameters:", best_params_boruta)

    # Step 5: Evaluate final model
    thresholds = evaluate_model(best_model_boruta, X_test_sel, y_test, output_file=output_path)

    # Step 6: SHAP analysis
    print("Running SHAP analysis...")
    run_shap(best_model_boruta, X_train_sel, X_test_sel)


    # Step 7: Save model
    model_package = {
        'model': best_model_boruta,
        'features': selected_features  
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(model_package, f)
    print("Model package saved successfully as model.pkl")


    return best_model_boruta, selected_features, thresholds



def main():

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

    # Run the pipeline
    run_pipeline(df)



