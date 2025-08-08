import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("/Users/farhat/Documents/Project/ProcessedData/fullData.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,[0,-1]]
X.head()
# Assuming you have X (features) and y (target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbose': -1
}

# Train the model
model = lgb.train(params, train_data, num_boost_round=50)
y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int)