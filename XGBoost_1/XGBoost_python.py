import xgboost as xgb
import numpy as np
import matplotlib.pylab as plt
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from display_data import import_data

#We dont want our model to care about the id of the house or the seller
data, data_test = import_data()
Y = data.price
data = data.drop(columns=['price','id','seller'])
categorical_data = ['layout', 'windows_court', 'windows_street', 'condition', 'building_id','new','district','street',
                    'address', 'material', 'elevator_without', 'elevator_passenger', 'elevator_service', 'parking','garbage_chute', 'heating']
data = data.drop(columns = categorical_data)
data_test = data_test.drop(columns = categorical_data)

for column in data.columns:
    column_type = data[column].dtype
    if column_type == 'object':
        break
    data[column] = data[column].replace(np.NaN, data[column].mean())

X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=42)

def root_mean_squared_log_error(y_true, y_pred):
    # Alternatively: sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5
    assert (y_true >= 0).all()
    assert (y_pred >= 0).all()
    log_error = np.log1p(y_pred) - np.log1p(y_true)  # Note: log1p(x) = log(1 + x)
    return np.mean(log_error ** 2) ** 0.5

data.info()
print(len(Y))
print(data.corr())

XGB_model = xgb.XGBRegressor(n_estimators = 10000)

XGB_model.fit(X_train,y_train)
prediction = XGB_model.predict(X_test)

count = 0
for i in prediction:
    if i < 0:
        prediction[count] = prediction.mean()
    count += 1

rmsle = root_mean_squared_log_error(y_test,prediction)
print("first run", rmsle)
