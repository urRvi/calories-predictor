import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv('../data/calories.csv')
    
    # Drop User_ID as it's not a feature
    if 'User_ID' in df.columns:
        df = df.drop('User_ID', axis=1)
        
    # Handle categorical variables
    # Check for object columns
    object_cols = df.select_dtypes(include=['object']).columns
    print(f"Categorical columns: {list(object_cols)}")
    
    label_encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    # Save label encoders for later use
    os.makedirs('src/models', exist_ok=True)
    joblib.dump(label_encoders, 'src/models/label_encoders.pkl')
    
    # Split features and target
    X = df.drop('Calories', axis=1)
    y = df['Calories']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'src/models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_xgboost(X_train, X_test, y_train, y_test):
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"XGBoost Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # Save model
    model.save_model('src/models/xgboost_model.json')
    return model, r2

class CaloriesNet(nn.Module):
    def __init__(self, input_dim):
        super(CaloriesNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_pytorch(X_train, X_test, y_train, y_test):
    print("\nTraining PyTorch model...")
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = CaloriesNet(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()
        
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"PyTorch Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'src/models/pytorch_model.pth')
    return model, r2

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    xgb_model, xgb_r2 = train_xgboost(X_train, X_test, y_train, y_test)
    pt_model, pt_r2 = train_pytorch(X_train, X_test, y_train, y_test)
    
    print("\nComparison:")
    print(f"XGBoost R2: {xgb_r2:.4f}")
    print(f"PyTorch R2: {pt_r2:.4f}")
    
    if xgb_r2 > pt_r2:
        print("XGBoost performed better.")
    else:
        print("PyTorch performed better.")
