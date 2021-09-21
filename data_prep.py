import numpy as np

# convert an array of values into a dataset matrix
def create_train_data(training_set_scaled, time_step, n_days_ahead):
    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    # print(training_set_scaled)
    for i in range(time_step, len(training_set_scaled)):
        try:
            output = training_set_scaled[i+n_days_ahead, 0]
            X_train.append(training_set_scaled[i - time_step:i, 0])
            y_train.append(output)
        except IndexError:
            break

    X_train, y_train = np.array(X_train), np.array(y_train)
    # print(X_train.shape)
    # print(y_train.shape)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train

def create_test_data(dataset_train, dataset_test, timestep, n_days_ahead,scaler):
    dataset_total = np.concatenate((dataset_train, dataset_test))
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestep:]
    inputs = inputs.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)
    X_test = []
    y_test = []
    # print("inputs", inputs)
    for i in range(timestep, len(dataset_test) + timestep):
        try:
            output = inputs[i+n_days_ahead, 0]
            X_test.append(inputs_scaled[i - timestep:i, 0])
            y_test.append(output)
        except IndexError:
            X_test.append(inputs_scaled[i - timestep:i, 0])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
    return X_test, inputs,y_test