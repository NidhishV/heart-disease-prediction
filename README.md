# Heart Disease Prediction using Machine Learning

## Project Overview
This project implements and compares supervised machine learning models to predict heart disease presence using a synthetic clinical dataset inspired by real-world patterns. It is structured to demonstrate a complete ML pipeline, from data preprocessing and exploration to modeling, evaluation, and calibration.

The goal is to develop a robust predictive model that can assist healthcare practitioners in identifying potential heart disease cases using accessible patient health metrics.

Technical Requirements
	•	Language: Python 3.8+
	•	Development Environment: Jupyter Notebook
	•	Key Libraries:
	•	pandas, numpy – data processing
	•	matplotlib, seaborn – data visualization
	•	scikit-learn – model building, tuning, and evaluation
	•	warnings, calibration_curve, RocCurveDisplay – reliability metrics

## Methodology and Approach
The project follows a standard end-to-end machine learning workflow:

1. Problem Definition

Can we predict whether an individual has heart disease based on their clinical features?

2. Data Exploration & Cleaning
	•	Checked for missing values and dropped incomplete rows.
	•	Conducted EDA with visualizations for:
	•	Heart disease frequency by sex
	•	Age distribution
	•	Chest pain type vs. heart disease frequency
	•	Feature correlations using a heatmap

3. Modeling and Evaluation

Two models were selected for classification:
	•	Logistic Regression – interpretable and commonly used in clinical settings
	•	Random Forest Classifier – powerful ensemble model

Evaluation techniques included:
	•	Accuracy, Precision, Recall, F1-score
	•	ROC-AUC score and ROC curve plots
	•	Confusion Matrix
	•	Feature importance analysis
	•	Calibration curves to assess model confidence

4. Model Optimization

Hyperparameter tuning using:
	•	RandomizedSearchCV
	•	GridSearchCV

## Data Sources and Description
	•	Dataset Size: ~150,000 entries with 14 features
	•	Data Type: Fully synthetic data, generated to reflect patterns found in real-world clinical datasets.
	•	Target Variable: target (1 = heart disease, 0 = no disease)
 Feaatures: 
1. age: Age of the individual in years.
   
2. sex: Gender of the individual (1 = male, 0 = female)

3. cp- Chest-pain type: displays the type of chest-pain experienced by the individual using the following format :
  * 0 = typical angina
  * 1 = atypical angina
  * 2 = non — anginal pain
  * 3 = asymptotic

4. trestbps- Resting Blood Pressure: displays the resting blood pressure value of an individual in mmHg (unit). anything above 130-140 is typically cause for concern.

5. chol- Serum Cholestrol: displays the serum cholesterol in mg/dl (unit)

6. fbs- Fasting Blood Sugar: compares the fasting blood sugar value of an individual with 120mg/dl.
  * If fasting blood sugar > 120mg/dl then : 1 (true) else : 0 (false) '>126' mg/dL signals diabetes

7.  restecg- Resting ECG : displays resting electrocardiographic results
  * 0 = normal
  * 1 = having ST-T wave abnormality
  * 2 = left ventricular hyperthrophy
  
8. thalach- Max heart rate achieved : displays the max heart rate achieved by an individual.

9. exang- Exercise induced angina : 1 = yes 0 = no

10. oldpeak- ST depression induced by exercise relative to rest: displays the value which is an integer or float.

11. slope- Slope of the peak exercise ST segment :
  * 0 = upsloping: better heart rate with excercise (uncommon)
  * 1 = flat: minimal change (typical healthy heart)
  * 2 = downsloping: signs of unhealthy heart

12. ca- Number of major vessels (0–3) colored by flourosopy : displays the value as integer or float.

13. thal : Displays the thalassemia :
  * 1,3 = normal
  * 6 = fixed defect
  * 7 = reversible defect: no proper blood movement when excercising

14. target : Displays whether the individual is suffering from heart disease or not :
  1 = yes 0 = no

## Key Code Sections:
Data Loading & Cleaning: CSV imported, missing values handled

EDA: Histograms, bar plots, crosstabs, correlation matrix

Modeling:
	•	Baseline scoring for both models
	•	Overfitting analysis via train/test accuracy
	•	Hyperparameter tuning (manual, random, grid search)
 
Evaluation:
	•	classification_report(), confusion_matrix()
	•	roc_auc_score, RocCurveDisplay
	•	calibration_curve for reliability
 
Interpretation:
	•	Feature importance via coef_ (LogReg) and feature_importances_ (RF)
	•	Bar plots for model insights

## Limitations and Future Work

Limitations:
	•	Synthetic data limits clinical applicability
	•	Random Forest model overfitting needs better generalization
	•	Lacks external validation on a real test cohort

Future Improvements:
	•	Introduce class balancing techniques
	•	Add more advanced models like XGBoost
