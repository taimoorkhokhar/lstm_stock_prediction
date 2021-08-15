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
output_file_path = 'performance/test_data_performance.csv'
data_path = 'stock_data/done_data.csv'
test_per = 10
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
    # print(X_train)
    # print(y_train)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train


def create_test_data(dataset_train, dataset_test, timestep):
    dataset_total = np.concatenate((dataset_train, dataset_test))
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestep:]
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test = []
    for i in range(timestep, len(dataset_test) + timestep):
        X_test.append(inputs[i - timestep:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test, inputs


def feature_scaling(df):
    df_scaled = scaler.fit_transform(df)
    return df_scaled


def model_prediction(model, X_test, X_train):
    ### Lets Do the prediction and check performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    ##Transformback to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    return train_predict, test_predict


def predict_future_days_price(test_data, time_steps, lstm_model):
    index = len(test_data)- time_steps
    previous_data_points = test_data[index:]
    previous_data_points_reshaped = test_data[index:].reshape(1,-1)
    temp_input = list(previous_data_points_reshaped)
    temp_input = temp_input[0].tolist()
    lst_output = []
    i = 0
    while i < FUTURE_DAYS_PREDICTION:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, time_steps, 1))
            y_predict = lstm_model.predict(x_input, verbose=0)
            temp_input.extend(y_predict[0].tolist())
            temp_input = temp_input[1:]
        else:
            x_input = previous_data_points_reshaped.reshape((1,time_steps, 1))
            y_predict = lstm_model.predict(x_input, verbose=0)
            temp_input.extend(y_predict[0].tolist())

        lst_output.extend(y_predict.tolist())
        i = i + 1

    future_predictions = scaler.inverse_transform(lst_output)
    deviation = np.std(future_predictions, dtype = np.float64)
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


def plot_results(dates, real_stock_price, predicted_stock_price, stock_name, future_predictions, future_dates):
    ### Plotting
    fig_handle, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
    ax.plot(dates, real_stock_price, color='black')
    future_dates=np.array(pd.to_datetime(future_dates, format='%Y%m%d'))
    ax.plot(dates,predicted_stock_price,'green',future_dates, future_predictions, 'red')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.tight_layout()
    plt.legend(('Stock Price','Predicted Stock Price','Future '+str(FUTURE_DAYS_PREDICTION)+' days Prediction'))
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.xaxis.set_minor_locator(mondays)
    ax.autoscale_view()
    ax.grid(True)
    fig_handle.autofmt_xdate()
    plt.savefig(f"plots/{stock_name}/prediction_plot.png")
    plt.close(fig_handle)


def dump_data_to_csv(data):
    keys = data[0].keys()
    with open(output_file_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)


def calculate_residual_error(true_values, predicted_values,stock_path):
    residual_error = np.subtract(true_values, predicted_values)
    # print(true_values, predicted_values)
    residuals = pd.DataFrame(residual_error.flatten(), columns=['errors'])
    error_statistics = residuals.describe()
    # print(error_statistics)
    # print(error_statistics.loc['mean'])
    # # summary statistics
    # print(residuals.describe())
    # print(type(residuals.describe()))
    # print(residuals)
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


    # fig, axes = plt.subplots(figsize=(12, 9))
    fig = sm.qqplot(residuals)
    fig.suptitle('Test Error QQ Plot', fontsize=12)
    plt.savefig(f"{stock_path}/error_qq_plot.png")
    plt.close(fig)
    error_max = error_statistics.loc['max'].values[0]
    error_min = error_statistics.loc['min'].values[0]
    error_mean = error_statistics.loc['mean'].values[0]
    error_std = error_statistics.loc['std'].values[0]
    error_median = error_statistics.loc['50%'].values[0]
    # df = pd.DataFrame(dict(min=error_min, max=error_max, mean=error_mean, std=error_std))
    fig = plt.figure(figsize=(9, 6))   
    plt.boxplot(residuals['errors'], showmeans=True)
    # plt.text(3, 0.07,
    #      str(error_mean),
    #      bbox=dict(facecolor='red',
    #                alpha=0.5),
    #      fontsize=12)
    triangle = mlines.Line2D([], [], color='mediumseagreen', marker='^', linestyle='None',
                          markersize=10, label='Mean (%.3f)' %error_mean)
    line = mlines.Line2D([], [], color='orange', marker='_', linestyle='None',
                          markersize=10, label='Median (%.3f)' %error_median)

    plt.legend(handles=[line, triangle])
    # plt.legend()
    plt.savefig(f"{stock_path}/error_box_plot.png")
    plt.close(fig)
    # print(error_statistics.loc['mean'])
    return error_mean,error_std


def callculate_stock_weight(current_stock_value, stock_value):
    # print(current_stock_value)
    # print(stock_value)
    return ((current_stock_value-stock_value) / abs(stock_value)) * 100

if __name__ == '__main__':
    makedir(['plots', 'performance'])
    df = pd.read_csv(data_path, index_col=[0])
    stock_model_performance = []
    # group the data on stocks
    stock_groups = df.groupby('tic')
    stock_count = 0 
    for name, group in stock_groups:
        makedir([f'plots/{name}'])
        stock_count += 1
        filtered_data = np.array(group['adjcp'].values)
        train_set, test_set = train_and_test_split(filtered_data)
        # prepare training data
        train_set_reshaped = train_set.reshape(-1, 1)
        train_set_scaled = feature_scaling(train_set_reshaped)
        time_step = 100
        X_train, y_train = create_train_data(train_set_scaled, time_step)
        model = train_lstm(X_train, y_train)
        # prepare test data
        X_test, test_set_scaled = create_test_data(train_set, test_set, time_step)
        # model prediction
        train_predict, test_predict = model_prediction(model, X_test, X_train)
        ### Test Data Standard Deviation
        test_dev = np.std(test_predict, dtype = np.float64)
        ### Test Data RMSE
        mse = mean_squared_error(test_set, test_predict)
        rmse = math.sqrt(mse)
        ### Test Data MAE
        mae = mean_absolute_error(test_set, test_predict)


        filtered_dates = np.array(pd.to_datetime(group['datadate'], format='%Y%m%d'))
        filtered_dates = filtered_dates[len(train_set):len(group)]
        future_predictions, deviation = predict_future_days_price(test_set_scaled, time_step, model)
        future_dates = generate_dates(filtered_dates[-1])
        future_dates = np.array(future_dates[1:])
        # print(filtered_dates[-1], "  ", future_dates[0])
        plot_results(filtered_dates, test_set, test_predict, name, future_predictions, future_dates)
        mean_forecast_error, std_forecast_error  = calculate_residual_error(test_set, test_predict,f'plots/{name}')
        # print(future_predictions)
        stock_weight = callculate_stock_weight(future_predictions[-1],test_predict[-1])
        performance = {
                        'stock name': name,
                        'MSE (mean square error)': mse,
                        'RMSE (root mean square error)': rmse,
                        'MAE (mean absolute error)' : mae,
                        'Mean Forecast Error' : mean_forecast_error,
                        'Standard Deviation Forecast Error' : std_forecast_error,
                        'stock_weight' : stock_weight[0]
                    }
        stock_model_performance.append(performance)
        # break
        if stock_count == 50:
            break
    dump_data_to_csv(stock_model_performance)
