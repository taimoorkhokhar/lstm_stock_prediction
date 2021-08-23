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
import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
from matplotlib.dates import MONDAY
# from statsmodels.graphics.gofplots import qqplots
import statsmodels.api as sm
import matplotlib.lines as mlines


# every monday
mondays = WeekdayLocator(MONDAY)
# every 3rd month
months = MonthLocator(range(1, 13), bymonthday=1, interval=1)
monthsFmt = DateFormatter("%b '%y")


FUTURE_DAYS_PREDICTION = 30
output_file_path = 'performance/'
data_path = 'stock_data/done_data.csv'
test_per = 10
scaler = MinMaxScaler(feature_range=(0, 1))
prediction_days_ahead= {
                        '1_day':1,
                        '3_days':3,
                        '1_week':7,
                        '1_month':30 
                        }

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
    model.add(Dense(y_train.shape[1]))

    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the Training set
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
    return model,history


def train_and_test_split(df):
    test_size = int(len(df) * (test_per / 100))
    train_size = len(df) - test_size
    training_data, test_data = df[0:train_size], df[train_size:len(df)]
    return training_data, test_data


# convert an array of values into a dataset matrix
def create_train_data(training_set_scaled, time_step, n_days_ahead):
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(time_step, len(training_set_scaled)):
        # print(i - time_step,i)
        output = training_set_scaled[i:i+n_days_ahead, 0]
        if len(output) ==  n_days_ahead:
            X_train.append(training_set_scaled[i - time_step:i, 0])
            y_train.append(output)

    X_train, y_train = np.array(X_train), np.array(y_train)
    # print(X_train.shape)
    # print(y_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train


def create_test_data(dataset_train, dataset_test, timestep, n_days_ahead):
    dataset_total = np.concatenate((dataset_train, dataset_test))
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestep:]
    inputs = inputs.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)
    X_test = []
    y_test = []
    # print("inputs", inputs)
    for i in range(timestep, len(dataset_test) + timestep):
        # print(i - timestep,i)
        output = inputs[i:i+n_days_ahead, 0]
        if len(output) == n_days_ahead:
            X_test.append(inputs_scaled[i - timestep:i, 0])
            y_test.append(output)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
    # print(y_test)
    return X_test, inputs,y_test


def feature_scaling(df):
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def model_prediction(model, X_test, X_train):
    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    ##Transformback to original form
    # print("test_predict ", X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    # print("test_predict ", test_predict)
    return train_predict, test_predict


def predict_future_days_price(test_data, time_steps, lstm_model,n_days_ahead):
    index = len(test_data)- time_steps
    previous_data_points = test_data[index:]
    previous_data_points_reshaped = test_data[index:].reshape(1,-1)
    temp_input = list(previous_data_points_reshaped)
    temp_input = temp_input[0].tolist()
    lst_output = []
    i = 0
    while i < FUTURE_DAYS_PREDICTION//n_days_ahead:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[-time_steps:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, time_steps, 1))
            y_predict = lstm_model.predict(x_input, verbose=0)
            temp_input.extend(y_predict[0].tolist())
            temp_input = temp_input[-time_steps:]
        else:
            x_input = previous_data_points_reshaped.reshape((1,time_steps, 1))
            y_predict = lstm_model.predict(x_input, verbose=0)
            temp_input.extend(y_predict[0].tolist())
        lst_output.extend(y_predict[0].tolist())
        i = i + 1
    lst_output = np.array(lst_output).reshape(1,-1)
    future_predictions = scaler.inverse_transform(lst_output)
    deviation = np.std(future_predictions, dtype = np.float64)
    print(future_predictions)
    print('future_predictions',len(future_predictions))
    # exit()
    return future_predictions, deviation


def generate_dates(start_date):
    start_date = pd.to_datetime(start_date)
    start_date = str(start_date.date())
    date_1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=FUTURE_DAYS_PREDICTION)
    end_date = end_date.strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date,end=end_date).to_pydatetime().tolist()
    formated_dates = [_date.strftime("%Y%m%d") for _date in dates]
    return formated_dates	


def plot_results(dates, real_stock_price, predicted_stock_price, stock_name):
    ### Plotting
    fig_handle, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    ax.plot(dates, real_stock_price, color='black')
    # future_dates=np.array(pd.to_datetime(future_dates, format='%Y%m%d'))
    ax.plot(dates,predicted_stock_price,'green')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.tight_layout()
    plt.legend(('Stock Price','Predicted Stock Price',))
    # plt.legend(('Stock Price','Predicted Stock Price','Future '+str(FUTURE_DAYS_PREDICTION)+' days Prediction'))
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(mondays)
    ax.autoscale_view()
    ax.grid(True)
    fig_handle.autofmt_xdate()
    plt.savefig(f"{stock_name}/prediction_plot.png")
    plt.close(fig_handle)
    plt.clf()


def dump_data_to_csv(data,csv_file_full_path):
    keys = data[0].keys()
    with open(csv_file_full_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


def calculate_residual_error(true_values, predicted_values,stock_path):
    residual_error = np.subtract(true_values, predicted_values)
    # print(true_values, predicted_values)
    residuals = pd.DataFrame(residual_error.flatten(), columns=['errors'])
    error_statistics = residuals.describe()

    fig, axes = plt.subplots(figsize=(12, 9))
    axes.plot(residuals)
    fig.suptitle('Test Error Line Plot')
    plt.savefig(f"{stock_path}/error_line_plot.png")
    plt.close(fig)

    fig= plt.figure(figsize=(12, 9))
    residuals.plot(kind='kde')
    plt.title("Test Error Density Plot")
    plt.savefig(f"{stock_path}/error_density_plot.png")
    plt.close(fig)

    fig = sm.qqplot(residuals)
    fig.suptitle('Test Error QQ Plot', fontsize=12)
    plt.savefig(f"{stock_path}/error_qq_plot.png")
    plt.close(fig)
    error_max = error_statistics.loc['max'].values[0]
    error_min = error_statistics.loc['min'].values[0]
    error_mean = error_statistics.loc['mean'].values[0]
    error_std = error_statistics.loc['std'].values[0]
    error_median = error_statistics.loc['50%'].values[0]
    fig = plt.figure(figsize=(9, 6))   
    plt.boxplot(residuals['errors'], showmeans=True)
    triangle = mlines.Line2D([], [], color='mediumseagreen', marker='^', linestyle='None',
                          markersize=10, label='Mean (%.3f)' %error_mean)
    line = mlines.Line2D([], [], color='orange', marker='_', linestyle='None',
                          markersize=10, label='Median (%.3f)' %error_median)

    plt.legend(handles=[line, triangle])
    # plt.legend()
    plt.savefig(f"{stock_path}/error_box_plot.png")
    plt.close(fig)

    plt.clf()
    # print(error_statistics.loc['mean'])
    return error_mean,error_std


def callculate_stock_weight(current_stock_value, stock_value):
    return ((current_stock_value-stock_value) / abs(stock_value)) * 100

def line_plot_for_multistep(rmse, n_days,  stock_plot_path):
    days = [f'Day {i+1}' for i in range(n_days)]
    fig_handle, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    ax.plot(days, rmse)
    plt.xlabel("Days")
    plt.ylabel("RMSE")
    plt.tight_layout()
    ax.legend(('Line Plot of RMSE for ' + str(n_days)+ ' days ahead prediction',))
    fig_handle.autofmt_xdate()
    plt.savefig(f"{stock_plot_path}/line_plot_rmse.png")
    plt.close(fig_handle)
    plt.clf()


def calculate_error_and_plot(test, forecasts, n_seq, stock_plot_path,test_data_dates):
    rmse=[]
    mae = []
    mse = []
    mfe = []
    sfe = []
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        if n_seq != 1:
            day_plot_path = f'{stock_plot_path}/{i+1}_day'
        else:
            day_plot_path = f'{stock_plot_path}'
        makedir([day_plot_path])
        mse_for_day = mean_squared_error(actual, predicted)
        mse.append(mse_for_day)
        rmse.append(math.sqrt(mse_for_day))
        mae.append(mean_absolute_error(actual, predicted))
        mean_forecast_error, std_forecast_error = calculate_residual_error(actual, predicted,day_plot_path)
        mfe.append(mean_forecast_error)
        sfe.append(std_forecast_error)
        # if test_data_dates > len(predicted)
        plot_results(test_data_dates, actual, predicted, day_plot_path)
    s = 0
    test = np.array(test)
    forecasts = np.array(forecasts)
    # print(test.shape)
    for row in range(test.shape[0]):
        for col in range(test.shape[1]):
            s += (test[row, col] - forecasts[row, col])**2
    overall_rmse = math.sqrt(s / (test.shape[0] * test.shape[1]))
    # print("score", overall_rmse)
    if n_seq >1:
        line_plot_for_multistep(rmse, n_seq,  stock_plot_path)
    return overall_rmse, rmse, mae, mse,mfe,sfe

if __name__ == '__main__':
    makedir(['plots', 'performance'])
    df = pd.read_csv(data_path, index_col=[0])
    # group the data on stocks
    stock_groups = df.groupby('tic')
    for days_ahead_key, n_days_ahead in prediction_days_ahead.items():
        stock_count = 0
        stock_model_performance = []
        for name, group in stock_groups:
            stock_count += 1
            stock_plot_path = f'plots/{days_ahead_key}_ahead/{name}'
            makedir([stock_plot_path])
            filtered_data = np.array(group['adjcp'].values)
            train_set, test_set = train_and_test_split(filtered_data)
            # prepare training data
            train_set_reshaped = train_set.reshape(-1, 1)
            train_set_scaled = feature_scaling(train_set_reshaped)
            time_step = 100
            X_train, y_train = create_train_data(train_set_scaled, time_step,n_days_ahead)
            model, history = train_lstm(X_train, y_train)
            training_loss = history.history.get("loss", [0])[-1]
            # print("training_loss", training_loss)
            # prepare test data
            X_test, test_set_scaled , y_test= create_test_data(train_set, test_set, time_step,n_days_ahead)
            # model prediction
            train_predict, test_predict = model_prediction(model, X_test, X_train)
            filtered_dates = np.array(pd.to_datetime(group['datadate'], format='%Y%m%d'))
            test_data_dates = filtered_dates[len(train_set):len(group)]
            test_data_dates = filtered_dates[:test_predict.shape[0]]
            # print(X_test.shape)
            # print(test_predict.shape)
            # print(len(test_data_dates))
            overall_rmse , rmse, mae, mse,mfe,sfe = calculate_error_and_plot(y_test, test_predict, n_days_ahead, stock_plot_path,test_data_dates)
            performance = {
                            'stock name': name,
                            'training_loss': training_loss
                        }
            for i in range(n_days_ahead):
                if n_days_ahead != 1:
                    text = f'{i+1} Day'
                    performance.update({f"overall RMSE":overall_rmse})
                else:
                    text = ''
                performance.update({f"{text} (RMSE (root mean square error))":rmse[i]})
                performance.update({f"{text} (MSE (mean square error))":mse[i]})
                performance.update({f"{text} (MAE (mean absolute error))":mae[i]})
                performance.update({f"{text} (Mean Forecast Error)":mfe[i]})
                performance.update({f"{text} (Standard Deviation Forecast Error)":sfe[i]})

            stock_model_performance.append(performance)
            # break
            if stock_count == 50:
                break
        csv_file_full_path =  f'{output_file_path}test_data_performance({days_ahead_key}_ahead).csv'
        dump_data_to_csv(stock_model_performance,csv_file_full_path)
