import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import math
from sklearn.metrics import mean_squared_error
import matplotlib
import csv
import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
from matplotlib.dates import MONDAY
import statsmodels.api as sm
import matplotlib.lines as mlines
import copy
from data_prep import create_train_data, create_test_data

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
                         '1_day':1
                        ,'3_days':3
                        ,'1_week':7
                        ,'1_month':30
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
    model.add(Dense(units=1))

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

def feature_scaling(df):
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def model_prediction(model, X_test, X_train):
    ### Lets Do the prediction and check performance metrics
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    ##Transformback to original form
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    return train_pred, test_pred


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
    # print(future_predictions)
    # print('future_predictions',len(future_predictions))
    # exit()
    return future_predictions, deviation


def generate_dates(start_date, days_ahead):
    start_date = pd.to_datetime(start_date)
    start_date = str(start_date.date())
    date_1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = date_1 + datetime.timedelta(days=days_ahead)
    end_date = end_date.strftime("%Y-%m-%d")
    # generate buisnes dates
    dates = pd.bdate_range(start=start_date,periods=days_ahead+1).to_pydatetime().tolist()
    formated_dates = [_date.strftime("%Y%m%d") for _date in dates]
    return np.array(pd.to_datetime(formated_dates[1:] , format='%Y%m%d'))


def plot_results(dates, real_stock_price, predicted_stock_price, stock_name):
    ### Plotting
    fig_handle, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    ax.plot(dates[:len(real_stock_price)], real_stock_price, color='black')
    # future_dates=np.array(pd.to_datetime(future_dates, format='%Y%m%d'))
    ax.plot(dates[len(dates)-len(predicted_stock_price):],predicted_stock_price,'green')
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
    plt.savefig(f"{stock_path}/error_box_plot.png")
    plt.close(fig)
    plt.clf()
    return error_mean,error_std


def callculate_stock_weight(current_stock_value, stock_value):
    return ((current_stock_value-stock_value) / abs(stock_value)) * 100



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
            # prepare test data
            X_test, test_set_scaled , y_test= create_test_data(train_set, test_set, time_step,n_days_ahead, scaler)
            # model prediction
            train_predict, test_predict_complete = model_prediction(model, X_test, X_train)
            filtered_dates = np.array(pd.to_datetime(group['datadate'], format='%Y%m%d'))
            test_data_dates = filtered_dates[len(train_set):len(group)]
            test_filtered_data = filtered_data[len(train_set):len(group)]
            test_predict = test_predict_complete[:len(y_test)]
            ### Test Data RMSE
            test_dev = np.std(test_predict, dtype=np.float64)
            mse = mean_squared_error(y_test, test_predict)
            rmse = math.sqrt(mse)
            ### Test Data MAE
            mae = mean_absolute_error(y_test, test_predict)
            mean_forecast_error, std_forecast_error = calculate_residual_error(y_test, test_predict, f'{stock_plot_path}')
            dates_for_the_day = copy.deepcopy(test_data_dates)
            n_dates_to_generate = len(test_predict_complete)- len(y_test)
            additional_dates_for_the_day = generate_dates(dates_for_the_day[-1],n_dates_to_generate)
            dates_for_the_day = np.append(dates_for_the_day, additional_dates_for_the_day)
            plot_results(dates_for_the_day, test_set, test_predict_complete, f'{stock_plot_path}')
            performance = {
                'stock name': name,
                'MSE (mean square error)': mse,
                'RMSE (root mean square error)': rmse,
                'MAE (mean absolute error)': mae,
                'Mean Forecast Error': mean_forecast_error,
                'Standard Deviation Forecast Error': std_forecast_error
            }


            stock_model_performance.append(performance)
            # break
            if stock_count == 50:
                break
        csv_file_full_path =  f'{output_file_path}test_data_performance({days_ahead_key}_ahead).csv'
        dump_data_to_csv(stock_model_performance,csv_file_full_path)
