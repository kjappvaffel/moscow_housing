import numpy as np
import matplotlib.pylab as plt
plt.style.use('ggplot')
#from feature_engineering import add_retning
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
#sys.path.append('../data')
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler


from display_data import import_data

def add_radius(df):
    #adds radius column to dataframe
    df['radius'] = np.sqrt((df['latitude']-55.75)**2 + (df['longitude']-37.55)**2)
    return df


def fix_geo_data(data_test):
    """should just be called on test data"""
    #fix all radius issues
    data_test._set_value(23,'longitude',37.473761)
    data_test._set_value(23,'latitude',55.560891)
    data_test._set_value(90,'longitude',37.473761)
    data_test._set_value(90,'latitude',55.560891)

    data_test._set_value(2511,'longitude',37.478055)
    data_test._set_value(2511,'latitude',55.544046)
    data_test._set_value(5090,'longitude',37.478055)
    data_test._set_value(5090,'latitude',55.544046)
    data_test._set_value(6959,'longitude',37.478055)
    data_test._set_value(6959,'latitude',55.544046)
    data_test._set_value(8596,'longitude',37.478055)
    data_test._set_value(8596,'latitude',55.544046)

    data_test._set_value(4719,'longitude',37.385493)
    data_test._set_value(4719,'latitude',55.853117)

    data_test._set_value(9547,'longitude',37.384711)
    data_test._set_value(9547,'latitude',55.853511)

    data_test._set_value(2529,'longitude',37.464994)
    data_test._set_value(2529,'latitude',55.627666)

    data_test = add_radius(data_test)

    return data_test

def add_high_up(df):
    """add exponetital function to determine how high up a building is"""
    high_up = df.floor/df.stories
    high_up_exp = np.exp(high_up) - 1
    euler = np.exp(1)

    df['high_up'] = high_up_exp
    df['high_up'].where(df['high_up'] > euler, euler)


    return df

from sklearn.model_selection import train_test_split

#We dont want our model to care about the id of the house or the seller
#In my first run, i will replace missing values with the mean value

data, data_test = import_data()

def lr_data_prep(data,data_test):
    Y = data.price
    data_RR = pd.DataFrame()
    data_RR['area_total'] = data['area_total']
    data_test_RR = pd.DataFrame()
    data_test_RR['area_total'] = data_test['area_total']
    print("gunnar")
    data = add_radius(data)
    data_RR['radius'] = data['radius']
    data_test = add_radius(data_test)
    data_test_RR['radius'] = data_test['radius']
    
    for column in data_RR:
        mean = data_RR[column].mean()
        mean_test = data_test_RR[column].mean()
        data_RR[column] = data_RR[column].replace(np.NaN, mean)
        data_test_RR[column] = data_test_RR[column].replace(np.NaN, mean_test)

    #data = add_high_up(data)
    #data_RR['high_up'] = data['high_up'] # Gjør dårligere


    #ceiling_mean = data['ceiling'].mean()
    #data['ceiling'] = data['ceiling'].replace(np.NaN, ceiling_mean)
    #data_RR['ceiling'] = data['ceiling'] ### GJør ikke bedre

    scaler = MinMaxScaler() # mapper alt til mellom 0 og 1, default
    data_RR = pd.DataFrame(scaler.fit_transform(data_RR))
    data_test_RR = pd.DataFrame(scaler.fit_transform(data_test_RR))
    
    print("styrk")
    return data_RR, data_test_RR

#y_plot = np.log(Y.values)/np.log(15)
#r_log_plot = np.log(data_RR['radius'].values*100)

#plt.plot(r_log_plot,y_plot)
#plt.show()
#plt.plot(data_RR['area_total'].values,y_plot)
#plt.show
#print(data_RR)
# add ceiling

