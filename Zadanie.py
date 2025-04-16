# Import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Wczytanie danych
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Czyszczenie danych
train_df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
train_df.dropna(subset=['Embarked'], inplace=True)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Usunięcie duplikatów
train_df.drop_duplicates(inplace=True)

# Podział cech i etykiety
X = train_df.drop(columns=['Survived'])
y = train_df['Survived']

# Wskazanie kolumn
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Preprocessing: pipeline numeryczny i kategoryczny
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Podział na dane treningowe i testowe
X_train_val, X_test_final, y_train_val, y_test_final = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modele do porównania
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

results_table = []

# Funkcja oceny modelu
def evaluate_model(name, model, X, y):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)
    row = {
        "Model": name,
        "Accuracy": np.mean(results['test_accuracy']),
        "Precision": np.mean(results['test_precision']),
        "Recall": np.mean(results['test_recall']),
        "F1 Score": np.mean(results['test_f1'])
    }
    results_table.append(row)

# Ocena wszystkich modeli
for name, model in models.items():
    evaluate_model(name, model, X_train_val, y_train_val)

# Tabela z wynikami
results_df = pd.DataFrame(results_table)
print("\nPorównanie modeli (średnie z walidacji krzyżowej):")
print(results_df)

# Wykres porównawczy (Accuracy + F1 Score)
results_df.set_index('Model')[['Accuracy', 'F1 Score']].plot(
    kind='bar', figsize=(10, 6), colormap='Set2'
)
plt.title("Porównanie modeli (Accuracy i F1 Score)")
plt.ylabel("Wynik")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.show()

# Finalny model bez SMOTE
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

final_pipeline.fit(X_train_val, y_train_val)
y_test_pred = final_pipeline.predict(X_test_final)

# Ocena bez SMOTE
print("\nEwaluacja na zbiorze TESTOWYM (Random Forest):")
print(classification_report(y_test_final, y_test_pred))

cm = confusion_matrix(y_test_final, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
plt.title("Macierz pomyłek (bez SMOTE)")
plt.show()

# Finalny model z SMOTE
smote_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

smote_pipeline.fit(X_train_val, y_train_val)
y_pred_smote = smote_pipeline.predict(X_test_final)

# Ocena z SMOTE
print("\nEwaluacja na zbiorze TESTOWYM z SMOTE (Random Forest):")
print(classification_report(y_test_final, y_pred_smote))

cm_smote = confusion_matrix(y_test_final, y_pred_smote)
ConfusionMatrixDisplay(confusion_matrix=cm_smote).plot(cmap='Blues')
plt.title("Macierz pomyłek (z SMOTE)")
plt.show()

# Porównanie wyników z i bez SMOTE
smote_scores = {
    "Model": "Random Forest + SMOTE",
    "Accuracy": accuracy_score(y_test_final, y_pred_smote),
    "Precision": precision_score(y_test_final, y_pred_smote),
    "Recall": recall_score(y_test_final, y_pred_smote),
    "F1 Score": f1_score(y_test_final, y_pred_smote)
}

baseline_scores = {
    "Model": "Random Forest (bez SMOTE)",
    "Accuracy": accuracy_score(y_test_final, y_test_pred),
    "Precision": precision_score(y_test_final, y_test_pred),
    "Recall": recall_score(y_test_final, y_test_pred),
    "F1 Score": f1_score(y_test_final, y_test_pred)
}

comparison_df = pd.DataFrame([baseline_scores, smote_scores])

# Wykres porównawczy SMOTE vs bez
comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(
    kind='bar', figsize=(10, 6), colormap='Accent'
)
plt.title("Porównanie metryk: Random Forest z SMOTE vs bez SMOTE")
plt.ylabel("Wynik")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
