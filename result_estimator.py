from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('shootouts.csv')

# Store original features for later display (for showing prediction examples)
X_original = df[['home_team', 'away_team', 'winner']].copy()

# Drop irrelevant columns for training
df_for_model = df.drop(columns=['date', 'first_shooter'])

# Define features (X) and target (y)
X = df_for_model[['home_team', 'away_team']]
y = df_for_model['winner']

# Encode categorical features
encoder_X = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder_X.fit_transform(X)

# Encode the target variable
encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# Get the indices for the split to reconstruct original data for test set examples
indices = np.arange(len(df_for_model))
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_encoded, y_encoded, indices, test_size=0.2, random_state=42
)

# Define the 4-layer feedforward neural network model
# This implies an input layer, two hidden layers (100 and 50 neurons respectively), and an output layer.
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42, verbose=True)

# Train the model
print("Training the neural network model...")
mlp.fit(X_train, y_train)
print("Training complete.")

# Make predictions on the test set
y_pred_encoded = mlp.predict(X_test)

# Decode predictions and actual labels to original team names
y_pred_decoded = encoder_y.inverse_transform(y_pred_encoded)
y_test_decoded = encoder_y.inverse_transform(y_test)

# Get the original home_team and away_team for the test set examples
X_test_original = X_original.iloc[indices_test]
X_test_original['predicted_winner'] = y_pred_decoded
X_test_original['actual_winner'] = y_test_decoded

# Display a few examples of predictions from the test set
print("\nテストセットからの予測例:")
print(X_test_original.to_string())

# Calculate and print overall accuracy
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"\nモデルの全体的な精度: {accuracy:.4f}")