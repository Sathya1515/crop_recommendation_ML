import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Check if 'label' column needs encoding
if data['label'].dtype == 'object':
    # Encode labels to integers
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])

# Split data into features (X) and labels (y)
X = data.drop('label', axis=1)  # Features (N, P, K, temperature, humidity, pH, rainfall)
y = data['label']                # Labels (crop types)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit PCA
pca = PCA(n_components=min(X_train_scaled.shape[1], len(y.unique()) - 1))
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize and fit LDA
lda = LDA(n_components=min(X_train_scaled.shape[1], len(y.unique()) - 1))
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# Initialize and train DecisionTreeClassifier for PCA
dt_classifier_pca = DecisionTreeClassifier(random_state=42)
dt_classifier_pca.fit(X_train_pca, y_train)

# Save the trained Decision Tree model to a file
joblib.dump(dt_classifier_pca, 'crop_recommendation_decision_tree_pca.joblib')

# Predict on the PCA test data
y_pred_pca = dt_classifier_pca.predict(X_test_pca)

# Calculate performance metrics for PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca, average='weighted')
recall_pca = recall_score(y_test, y_pred_pca, average='weighted')
f1_pca = f1_score(y_test, y_pred_pca, average='weighted')

print(f"PCA Accuracy: {accuracy_pca * 100:.2f}%")
print(f"PCA Precision: {precision_pca:.2f}")
print(f"PCA Recall: {recall_pca:.2f}")
print(f"PCA F1 Score: {f1_pca:.2f}")

# Print classification report for PCA
print("\nPCA Classification Report:\n")
print(classification_report(y_test, y_pred_pca))

# Initialize and train DecisionTreeClassifier for LDA
dt_classifier_lda = DecisionTreeClassifier(random_state=42)
dt_classifier_lda.fit(X_train_lda, y_train)

# Save the trained Decision Tree model to a file
joblib.dump(dt_classifier_lda, 'crop_recommendation_decision_tree_lda.joblib')

# Predict on the LDA test data
y_pred_lda = dt_classifier_lda.predict(X_test_lda)

# Calculate performance metrics for LDA
accuracy_lda = accuracy_score(y_test, y_pred_lda)
precision_lda = precision_score(y_test, y_pred_lda, average='weighted')
recall_lda = recall_score(y_test, y_pred_lda, average='weighted')
f1_lda = f1_score(y_test, y_pred_lda, average='weighted')

print(f"LDA Accuracy: {accuracy_lda * 100:.2f}%")
print(f"LDA Precision: {precision_lda:.2f}")
print(f"LDA Recall: {recall_lda:.2f}")
print(f"LDA F1 Score: {f1_lda:.2f}")

# Print classification report for LDA
print("\nLDA Classification Report:\n")
print(classification_report(y_test, y_pred_lda))

# Visualize PCA transformation
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis', s=60)
plt.title('PCA: First Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Crop Type')
plt.show()

# Visualize LDA transformation
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_train_lda[:, 0], y=X_train_lda[:, 1], hue=y_train, palette='viridis', s=60)
plt.title('LDA: First Two Linear Discriminants')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(title='Crop Type')
plt.show()
