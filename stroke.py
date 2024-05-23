import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import time

# Load the data
data = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(data.head())
print(data.info())
# Define the mapping
gender_mapping = {"Male": 0, "Female": 1,"Other":2}

# Apply the mapping to the "gender" column
data["gender"] = data["gender"].map(gender_mapping)
print(data.head())
print(data.info())
# Drop the 'id' column
data.drop(columns='id', axis=1, inplace=True)
print(data.head())
print(data.info())
# Handle missing values
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
print(data.head())
print(data.info())



from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Columns to encode
cols_to_encode = ['gender','work_type', 'Residence_type', 'smoking_status']

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns
for col in cols_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

# Encode 'ever_married' column
data['ever_married'] = label_encoder.fit_transform(data['ever_married'])

# Apply SMOTE
smote = SMOTE(sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(data.drop(columns=['stroke']), data['stroke'])

print(X_resampled.shape)
print(y_resampled.shape)

print(data.head())
print(data.info())

print(data.info())
cat_cols = ['gender','ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

    # Scale features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_resampled)
    print(X_std[:5])
    # Encode labels as integers starting with 0
    y = y_resampled.astype(int)
models = [
    ('xgb', XGBClassifier(objective="binary:logistic", use_label_encoder=False, eval_metric='error', random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42,force_row_wise=False, force_col_wise=True,min_child_samples=10)),

]
# Stacking Classifier
stacking_model = StackingClassifier(estimators=models, final_estimator=LogisticRegression())
# Hyperparameter tuning
param_grid = {
    'rf__n_estimators': [10, 38],
    'xgb__max_depth': [2, 3],
    'lgbm__num_leaves': [19, 10]
}
# Define early stopping criteria
early_stopping = {'accuracy': 0.98, 'tolerance': 5}  # Stop if validation accuracy reaches 99% or no improvement after 5 iterations

# Initialize lists to store accuracy and iteration data
accuracy_list = []
iteration_list = []

# Perform GridSearchCV with early stopping
best_accuracy = 0
tolerance_count = 0
max_iterations = 10
for i in range(max_iterations):
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_std, y, test_size=0.2, random_state=i, stratify=y)

    # Transform the training data
    X_train_scaled = scaler.transform(X_train)

    # Perform GridSearchCV
    grid_search = GridSearchCV(stacking_model, param_grid, cv=2, scoring='accuracy', verbose=0, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Validate the model
    val_accuracy = accuracy_score(y_val, best_model.predict(X_val))
    accuracy_list.append(val_accuracy)
    iteration_list.append(i + 1)

    # Check for early stopping
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        tolerance_count = 0
    else:
        tolerance_count += 1

    if best_accuracy >= early_stopping['accuracy'] or tolerance_count >= early_stopping['tolerance']:
        print(f"Early stopping criteria met. Best Validation Accuracy: {best_accuracy}")
        break

    # Plot learning curve for each iteration
    plt.plot(iteration_list, accuracy_list, marker='o')
    plt.title('Validation Accuracy vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()

    print(data.head())
    print(data.info())
    # Calculate remaining fits
remaining_fits = max_iterations - (i + 1)
print(f"\n ********************Remaining fits: {remaining_fits}*************************")

# Train final model on entire dataset
best_model.fit(X_std, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42, stratify=y)
print(X_test.shape)
# Fit StackingClassifier after hyperparameter tuning
stacking_model.fit(X_train, y_train)
print(data.head())
print(data.info())
# Predictions on test set
y_pred = best_model.predict(X_test)
print(y_pred)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n Accuracy on Test Set: {:.4f}".format(accuracy))

print(classification_report(y_test, y_pred))

end_time = time.time()
#print(f"\nTotal time taken: {end_time - start_time:.4f} seconds")
print(data.head())
print(data.info())
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# ROC AUC Score
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score: {:.4f}".format(auc_score))



