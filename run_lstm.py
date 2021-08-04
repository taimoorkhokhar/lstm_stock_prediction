import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
import csv
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
from matplotlib.dates import MONDAY
# every monday
mondays = WeekdayLocator(MONDAY)
# every 3rd month
months = MonthLocator(range(1, 13), bymonthday=1, interval=3)
monthsFmt = DateFormatter("%b '%y")


output_file_path = 'performance/lstm_performance.csv'
data_path = 'stock_data/done_data.csv'
test_per = 20
scaler = MinMaxScaler(feature_range=(0, 1))

def makedir(dirs_used):
    for _d in dirs_used:
        if not os.path.exists(_d):
            os.makedirs(_d)


def train_lstm(X_train, y_train):
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs=2, batch_size=64, verbose=1)
    return model


def train_and_test_split(df):
    test_size = int(len(df) * (test_per / 100))
    train_size = len(df) - test_size
    training_data, test_data = df[0:train_size], df[train_size:len(df)]
    return training_data, test_data


# convert an array of values into a dataset matrix
def create_train_data(training_set_scaled, time_step=1):
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(time_step, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - time_step:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train,y_train


def feature_scaling(df):
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def create_test_data(dataset_train, dataset_test, timestep):
    dataset_total = np.concatenate((dataset_train,dataset_test))
    inputs = dataset_total[len(dataset_total) - len(dataset_test) -  timestep:]
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(timestep, len(dataset_test)+timestep):
        X_test.append(inputs[i- timestep:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test


def model_prediction(model, X_test, X_train):
    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    ##Transformback to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    return train_predict, test_predict


def plot_results(dates, real_stock_price,predicted_stock_price, stock_name):
    ### Plotting
    fig_handle ,ax  = plt.subplots()
    ax.plot(dates, real_stock_price, color = 'black', label = 'Stock Price')
    ax.plot(dates, predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.tight_layout()
    plt.legend()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(mondays)
    ax.autoscale_view()
    ax.grid(True)
    fig_handle.autofmt_xdate()
    plt.savefig(f"plots/{stock_name}.png")


def dump_data_to_csv(data):
	keys = data[0].keys()
	with open(output_file_path, 'w', newline='') as output_file:
	    dict_writer = csv.DictWriter(output_file, keys)
	    dict_writer.writeheader()
	    dict_writer.writerows(data)



if __name__ == '__main__':
    makedir(['plots','performance'])
    df = pd.read_csv(data_path, index_col=[0])
    stock_model_performance = []
    # group the data on stocks
    stock_groups = df.groupby('tic')
    for name, group in stock_groups:
        filtered_data = np.array(group['adjcp'].values)
        train_set, test_set = train_and_test_split(filtered_data)
        # prepare training data
        train_set_reshaped = train_set.reshape(-1,1)
        # print(train_set_reshaped)
        train_set_scaled = feature_scaling(train_set_reshaped)
        time_step = 100
        X_train, y_train = create_train_data(train_set_scaled, time_step)
        model = train_lstm(X_train, y_train)
        # prepare test data
        X_test = create_test_data(train_set,test_set,time_step)
        # model prediction
        train_predict, test_predict = model_prediction(model, X_test, X_train)
        filtered_dates = np.array(pd.to_datetime(group['datadate'], format='%Y%m%d'))
        filtered_dates = filtered_dates[len(train_set):len(group)]
        ### Test Data RMSE
        rmse = math.sqrt(mean_squared_error(test_set,test_predict))
        print("rmse == ", rmse)
        plot_results(filtered_dates, test_set,test_predict, name)
        performance = {'stock_name': name, 
        				'RMSE (root mean square error)':rmse
        			}
        stock_model_performance.append(performance)

        # break
    dump_data_to_csv(stock_model_performance)
