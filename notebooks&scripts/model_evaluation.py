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

#----------------------------------------------------------------------------------------------------------------
# Load the model, input model path
with open("path/to/model.pkl", "rb") as f:
	loaded_package = pickle.load(f)

loaded_model = loaded_package['model']
loaded_features = loaded_package['features']
loaded_thresholds = loaded_package['optimal_thresholds']
loaded_classes = np.array(loaded_package['classes'])

# Print the full set of features 
print("features used in model training:", loaded_features)


# Load the validation set, input the actual file path
df_validation=pd.read_csv("path/to/data.csv")

columns_to_keep=loaded_features
X_test=df_validation[columns_to_keep]
y_test=df_validation[['CH']]

#-------------------------------------------------------------------------------------------------------------------
# ROC-AUC plot, AUC and threshold determination

# Function: ROC-AUC plot & thresholds
def plot_roc_auc(model, X, y, save_path="roc_auc.png"):
	"""
	Plot and save ROC curves for multi-class classification.
	Returns: dict with class-wise optimal thresholds and AUCs.
	"""
	classes = np.unique(y)
	y_bin = label_binarize(y, classes=classes)
	y_pred_proba = model.predict_proba(X)

	plt.figure(figsize=(5, 4))
	colors = ['blue', 'red', 'green', 'purple', 'orange']
	auc_scores = {}

	for i, cls in enumerate(classes):
		fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_pred_proba[:, i])
		roc_auc = auc(fpr, tpr)
		auc_scores[cls] = roc_auc

		plt.plot(
			fpr, tpr,
			color=colors[i % len(colors)],
			lw=2,
			label=f"Class {cls} (AUC={roc_auc:.2f})"
		)

	plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6)
	plt.xlabel("False Positive Rate", fontsize=12)
	plt.ylabel("True Positive Rate", fontsize=12)
	plt.title("Multi-Class ROC Curve", fontsize=14)
	plt.legend()
	plt.grid(alpha=0.3)
	plt.tight_layout()

	# Save and show
	plt.savefig(save_path, dpi=300)
	plt.show()

	return {"auc_scores": auc_scores, "proba": y_pred_proba}


#-------------------------------------------------------------------------------------------------------------------

# Function: Confusion matrix plotting & saving
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
	"""
	Plot and save a confusion matrix with a blue colormap.
	"""
	cm = confusion_matrix(y_true, y_pred)
	cmap = sns.color_palette("Blues", as_cmap=True)

	plt.figure(figsize=(5, 4))
	disp = ConfusionMatrixDisplay(confusion_matrix=cm)
	disp.plot(cmap=cmap, ax=plt.gca(), values_format='d', text_kw={"fontsize": 10})
	plt.title("Confusion Matrix", fontsize=14)
	plt.xlabel("Predicted Label", fontsize=12)
	plt.ylabel("True Label", fontsize=12)
	plt.tight_layout()

	# Save and show
	plt.savefig(save_path, dpi=300)
	plt.show()

# -------------------------------------------------------
# Post-NGS risk stratification performance

def save_chrs_statistics(test_df, y_pred, output_file="chrs_statistics.txt"):
	"""
	Compute CHRS-based and MN-based statistics for predicted labels and
	save the results to a text file.
	
	Parameters
	----------
	test_df : pd.DataFrame
		Must contain columns ['CHRS', 'MN'].
	y_pred : array-like
		Predicted labels (0, 1, or 2).
	output_file : str, optional
		Path of the text file to save results. Default is 'chrs_statistics.txt'.
	"""

	# --- 1. True CHRS risk distributions ---
	hirisk_true = (test_df['CHRS'] >= 12.5).sum()
	intermediate_true = ((test_df['CHRS'] >= 10) & (test_df['CHRS'] < 12.5)).sum()
	lowrisk_true = ((test_df['CHRS'] > 0) & (test_df['CHRS'] < 10)).sum()
	norisk_true = (test_df['CHRS'] == 0).sum()

	# --- 2. True CHRS with MN ---
	high_true_with_MN = ((test_df['CHRS'] >= 12.5) & (test_df['MN'] == 1)).sum()
	intermediate_true_with_MN = ((test_df['CHRS'] >= 10) & (test_df['CHRS'] < 12.5) & (test_df['MN'] == 1)).sum()

	# --- 3. Predicted classes ---
	mask_2 = (y_pred == 2)
	mask_1 = (y_pred == 1)
	mask_0 = (y_pred == 0)

	predicted_class2_df = test_df.iloc[np.where(mask_2)[0]]
	predicted_class1_df = test_df.iloc[np.where(mask_1)[0]]
	predicted_class0_df = test_df.iloc[np.where(mask_0)[0]]

	# --- 4. Function to get risk counts ---
	def get_risk_counts(df):
		return {
			"high": (df['CHRS'] >= 12.5).sum(),
			"intermediate": ((df['CHRS'] >= 10) & (df['CHRS'] < 12.5)).sum(),
			"low": ((df['CHRS'] > 0) & (df['CHRS'] < 10)).sum(),
			"none": (df['CHRS'] == 0).sum()
		}

	pred2 = get_risk_counts(predicted_class2_df)
	pred1 = get_risk_counts(predicted_class1_df)
	pred0 = get_risk_counts(predicted_class0_df)

	# --- 5. CHRS + MN combinations ---
	CHMN = ((test_df['CHRS'] > 0) & (test_df['MN'] == 1)).sum()
	CHnoMN = ((test_df['CHRS'] > 0) & (test_df['MN'] == 0)).sum()
	noCHMN = ((test_df['CHRS'] == 0) & (test_df['MN'] == 1)).sum()
	noCHnoMN = ((test_df['CHRS'] == 0) & (test_df['MN'] == 0)).sum()

	MN_pred2 = ((predicted_class2_df['CHRS'] > 0) & (predicted_class2_df['MN'] == 1)).sum()
	MN_pred1 = ((predicted_class1_df['CHRS'] > 0) & (predicted_class1_df['MN'] == 1)).sum()


	# --- NEW.  MN predictions ---
	MN = (test_df['MN'] == 1).sum()
	noMN = (test_df['MN'] == 0).sum()

	all_MN_pred2 = (predicted_class2_df['MN'] == 1).sum()
	all_MN_pred1 = (predicted_class1_df['MN'] == 1).sum()
	all_MN_pred0 = (predicted_class0_df['MN'] == 1).sum()

	print(all_MN_pred2,all_MN_pred1,all_MN_pred0)

	# --- 6. High and intermediate risk with MN among predictions ---
	high_pred2_with_MN = ((predicted_class2_df['CHRS'] >= 12.5) & (predicted_class2_df['MN'] == 1)).sum()
	high_pred1_with_MN = ((predicted_class1_df['CHRS'] >= 12.5) & (predicted_class1_df['MN'] == 1)).sum()
	intermediate_pred2_with_MN = ((predicted_class2_df['CHRS'] >= 10) & (predicted_class2_df['CHRS'] < 12.5) & (predicted_class2_df['MN'] == 1)).sum()
	intermediate_pred1_with_MN = ((predicted_class1_df['CHRS'] >= 10) & (predicted_class1_df['CHRS'] < 12.5) & (predicted_class1_df['MN'] == 1)).sum()

	# --- 7. Percentages ---
	pct_high_pred_in_1or2 = round((pred2["high"] + pred1["high"]) * 100 / hirisk_true, 2) if hirisk_true else 0
	pct_high_correct = round(pred2["high"] * 100 / hirisk_true, 2) if hirisk_true else 0
	pct_intermediate_pred_in_1or2 = round((pred2["intermediate"] + pred1["intermediate"]) * 100 / intermediate_true, 2) if intermediate_true else 0
	pct_intermediate_pred_in_0 = round(pred0["intermediate"] * 100 / intermediate_true, 2) if intermediate_true else 0

	# --- 8. Write results to file ---
	with open(output_file, "w") as f:
		f.write("=== CHRS AND MN STATISTICS ===\n\n")

		f.write("Actual Risk Distribution:\n")
		f.write(f"  High-risk: {hirisk_true}\n")
		f.write(f"  Intermediate-risk: {intermediate_true}\n")
		f.write(f"  Low-risk: {lowrisk_true}\n")
		f.write(f"  No-risk: {norisk_true}\n\n")

		f.write("Predicted Class Counts:\n")
		f.write(f"  Class 2 (CH+ with MN): {len(predicted_class2_df)}\n")
		f.write(f"  Class 1 (CH+ without MN): {len(predicted_class1_df)}\n")
		f.write(f"  Class 0 (Everyone else): {len(predicted_class0_df)}\n\n")

		f.write("Risk Distribution Among Predictions:\n")
		for label, counts in zip(["Class 2", "Class 1", "Class 0"], [pred2, pred1, pred0]):
			f.write(f"  {label}:\n")
			f.write(f"    High-risk: {counts['high']}\n")
			f.write(f"    Intermediate-risk: {counts['intermediate']}\n")
			f.write(f"    Low-risk: {counts['low']}\n")
			f.write(f"    No-risk: {counts['none']}\n\n")

		f.write("=== MN-SPECIFIC COUNTS ===\n")
		f.write(f"Actual high-risk with MN: {high_true_with_MN}\n")
		f.write(f"  Among predicted positives with MN (class 2): {high_pred2_with_MN}\n")
		f.write(f"  Among predicted positives without MN (class 1): {high_pred1_with_MN}\n\n")
		f.write(f"Actual intermediate-risk with MN: {intermediate_true_with_MN}\n")
		f.write(f"  Among predicted positives with MN (class 2): {intermediate_pred2_with_MN}\n")
		f.write(f"  Among predicted positives without MN (class 1): {intermediate_pred1_with_MN}\n\n")

		f.write("CHRS vs MN Distribution in Test Data:\n")
		f.write(f"  CH+ with MN: {CHMN}\n")
		f.write(f"  CH+ without MN: {CHnoMN}\n")
		f.write(f"  CH- with MN: {noCHMN}\n")
		f.write(f"  CH- without MN: {noCHnoMN}\n\n")

		f.write("Predicted CH+ with MN Individuals in Positive Classes:\n")
		f.write(f"  In class 2: {MN_pred2}\n")
		f.write(f"  In class 1: {MN_pred1}\n\n")

	print(f"✅ Results saved to '{output_file}'")

# -------------------------------------------------------
def main():


	# Run ROC-AUC and get thresholds
	roc_results = plot_roc_auc(loaded_model, X_test, y_test, save_path="roc_auc.png")
	y_proba = roc_results["proba"]

	# Create a vector of thresholds ordered by class index
	classes_sorted = np.unique(y_test)
	thresholds_array = np.array([loaded_thresholds[cls] for cls in classes_sorted])

	# Custom predictions using thresholds
	y_pred_custom = []
	for prob_vector in y_proba:
		passing = np.where(prob_vector >= thresholds_array)[0]
		if passing.size > 0:
			chosen = passing[np.argmax(prob_vector[passing])]
		else:
			chosen = np.argmax(prob_vector)
		y_pred_custom.append(classes_sorted[chosen])

	y_pred_custom = np.array(y_pred_custom)

	# -------------------------------------------------------
	# Accuracy and classification report
	accuracy = accuracy_score(y_test, y_pred_custom)
	print(f"\nAccuracy Score: {accuracy:.4f}\n")

	class_report = classification_report(y_test, y_pred_custom)
	print("Classification Report:\n", class_report)

	# Save classification report
	with open("classification_report.txt", "w") as f:
		f.write(f"Accuracy: {accuracy:.4f}\n\n")
		f.write(class_report)

	   # -------------------------------------------------------
	# Confusion matrix plot
	plot_confusion_matrix(y_test, y_pred_custom, save_path="confusion_matrix.png")

	# -------------------------------------------------------
	# Post-NGS risk stratification performance
	save_chrs_statistics(df_validation, y_pred_custom, output_file="chrs_statistics.txt")


if __name__ == "__main__":
	main()  
