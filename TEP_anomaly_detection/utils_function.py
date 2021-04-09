# created by LZ
# utils function for anomaly detection

import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt
from matplotlib import pyplot

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json

import timeseries_preprocessing as tp
from timeseries_errors import reconstruction_errors
import utils

from scipy import stats


def read_csv_data(root_folder, data_name_list):
    data_list = []
    for data_name in data_name_list:
        data_temp = pd.read_csv(root_folder + "data_set_csv/" + data_name + ".csv", sep=",", header=None, ).to_numpy()
        data_list.append(data_temp)
    return data_list


def data_standardization(root_folder, data_name_list, eta=2):
    data_list = read_csv_data(root_folder, data_name_list)

    # use fault_free_train_250000 to set the scaler
    data_index_set_scaler = data_name_list.index("fault_free_train_250000")
    data_set_scaler = data_list[data_index_set_scaler]
    scaler = StandardScaler().fit(data_set_scaler[:, 1:])

    data_standard_list = []
    for i in range(len(data_name_list)):
        data_standard_temp = scaler.transform(data_list[i][:, 1:]) / eta
        data_label_temp = np.reshape(data_list[i][:, 0], (data_list[i].shape[0], 1))
        data_standard_temp = np.hstack((data_label_temp, data_standard_temp))

        data_standard_list.append(data_standard_temp)
    return data_standard_list, scaler


# modified for multivariate version
def rolling_window_sequences_3D(X, index, window_size, target_size=1, step_size=1, target_column=0, offset=0, drop=None,
                                drop_windows=False):
    X_length = X.shape[0]
    numOfMultivariant = X.shape[1]

    X_ = np.reshape(X[:, 0], (X_length, 1))
    _, out_y, _, y_index = tp.rolling_window_sequences(X_, index, window_size,
                                                       target_size, step_size,
                                                       target_column, offset=offset,
                                                       drop=drop,
                                                       drop_windows=drop_windows)
    for m in range(1, numOfMultivariant):
        X_ = np.reshape(X[:, m], (X_length, 1))
        _, out_y_temp, _, _ = tp.rolling_window_sequences(X_, index, window_size,
                                                          target_size, step_size,
                                                          target_column, offset=offset,
                                                          drop=drop,
                                                          drop_windows=drop_windows)
        out_y = np.hstack((out_y, out_y_temp))

    out_X, _, X_index, _ = tp.rolling_window_sequences(X, index, window_size,
                                                       target_size, step_size,
                                                       target_column, offset=offset,
                                                       drop=drop,
                                                       drop_windows=drop_windows)
    return out_X, out_y, X_index, y_index


# modified for multivariate version
def unroll_ts_3D(X_hat):
    X_length = X_hat.shape[0]
    X_win = X_hat.shape[1]
    numOfMultivariant = X_hat.shape[2]

    X_hat_ = np.reshape(X_hat[:, :, 0], (X_length, X_win, 1))
    y = utils.unroll_ts(X_hat_)
    y = np.reshape(y, (y.shape[0], 1))

    for m in range(1, numOfMultivariant):
        X_hat_ = np.reshape(X_hat[:, :, m], (X_length, X_win, 1))
        y_temp = utils.unroll_ts(X_hat_)
        y_temp = np.reshape(y_temp, (y_temp.shape[0], 1))

        y = np.hstack((y, y_temp))
    return y


def pair_wise_error_detection(y, y_hat):
    # pair-wise error calculation
    error = np.zeros(shape=y.shape)
    length = y.shape[0]
    for i in range(length):
        error[i] = abs(y_hat[i] - y[i])

    # visualize the error curve
    fig = plt.figure(figsize=(30, 3))
    plt.plot(error)
    plt.show()


def compute_critic_score(critics, smooth_window):
    """Compute an array of anomaly scores.

    Args:
        critics (ndarray):
            Critic values.
        smooth_window (int):
            Smooth window that will be applied to compute smooth errors.

    Returns:
        ndarray:
            Array of anomaly scores.
    """
    critics = np.asarray(critics)
    l_quantile = np.quantile(critics, 0.25)
    u_quantile = np.quantile(critics, 0.75)
    in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
    critic_mean = np.mean(critics[in_range])
    critic_std = np.std(critics)

    z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
    z_scores = pd.Series(z_scores).rolling(
        smooth_window, center=True, min_periods=smooth_window // 2).mean().values

    return z_scores


def score_anomalies_3D(y, y_hat, critic, index, score_window=10, critic_smooth_window=None,
                       error_smooth_window=None, smooth=True, rec_error_type="point", comb="mult",
                       lambda_rec=0.5):
    """Compute an array of anomaly scores.

    Anomaly scores are calculated using a combination of reconstruction error and critic score.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        index (ndarray):
            time index for each y (start position of the window)
        critic (ndarray):
            Critic score. Each timestamp has multiple critic scores.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        critic_smooth_window (int):
            Optional. Size of window over which smoothing is applied to critic.
            If not given, 200 is used.
        error_smooth_window (int):
            Optional. Size of window over which smoothing is applied to error.
            If not given, 200 is used.
        smooth (bool):
            Optional. Indicates whether errors should be smoothed.
            If not given, `True` is used.
        rec_error_type (str):
            Optional. The method to compute reconstruction error. Can be one of
            `["point", "area", "dtw"]`. If not given, 'point' is used.
        comb (str):
            Optional. How to combine critic and reconstruction error. Can be one
            of `["mult", "sum", "rec"]`. If not given, 'mult' is used.
        lambda_rec (float):
            Optional. Used if `comb="sum"` as a lambda weighted sum to combine
            scores. If not given, 0.5 is used.

    Returns:
        ndarray:
            Array of anomaly scores.
    """

    critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0] * 0.01)
    error_smooth_window = error_smooth_window or math.trunc(y.shape[0] * 0.01)

    step_size = 1  # expected to be 1

    true_index = index  # no offset

    # commented out by LZ
    # need to be modified for multivariante
    # true = [item[0] for item in y.reshape((y.shape[0], -1))]
    # for item in y[-1][1:]:
    #     true.extend(item)

    # added by LZ
    # retrieve the true value from y (after rolling window operation)
    true = [item[0] for item in y]
    for item in y[-1][1:]:
        true.append(item)

    critic_extended = list()
    for c in critic:
        critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())

    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))
    print("critic_extended.shape: ", critic_extended.shape)

    critic_kde_max = []
    pred_length = y_hat.shape[1]

    # equals (number of points minus one)
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)

    for i in range(num_errors):
        critic_intermediate = []

        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            critic_intermediate.append(critic_extended[i - j, j])

        if len(critic_intermediate) > 1:
            discr_intermediate = np.asarray(critic_intermediate)
            try:
                critic_kde_max.append(discr_intermediate[np.argmax(
                    stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
            except np.linalg.LinAlgError:
                critic_kde_max.append(np.median(discr_intermediate))
        else:
            critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

    print("len(critic_kde_max): ", len(critic_kde_max))

    # Compute critic scores
    critic_scores = compute_critic_score(critic_kde_max, critic_smooth_window)

    print("critic_scores.shape: ", critic_scores.shape)

    # Compute reconstruction scores
    rec_scores, predictions = reconstruction_errors(
        y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)

    print("rec_scores.shape: ", rec_scores.shape)

    rec_scores = stats.zscore(rec_scores)
    print("rec_scores.shape: ", rec_scores.shape)

    rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1
    print("rec_scores.shape: ", rec_scores.shape)

    # Combine the two scores
    if comb == "mult":
        final_scores = np.multiply(critic_scores, rec_scores)
        print("final_score.shape", final_scores.shape)

    elif comb == "sum":
        final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)

    elif comb == "rec":
        final_scores = rec_scores

    else:
        raise ValueError('Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))

    # commented out by LZ
    # true = [[t] for t in true]

    return final_scores, true_index, true, predictions, critic_scores, rec_scores


# calculate the detection rate/false alarm for binary classification
def detect_binary_classification(data, data_pred, threshold, feature_list=list(range(52))):
    data_clf = detect_with_reconstruct_error(data, data_pred, threshold, feature_list)

    # detection_rate = np.count_nonzero(prediction) / prediction.shape[0]
    detection_rate = np.count_nonzero(data_clf) / data_clf.shape[0]
    # s1 = label + str(detection_rate)
    # print(s1)
    return detection_rate


# use the threshold to detect the anomaly of autoencoder using reconstruction error
def detect_with_reconstruct_error(data, data_pred, threshold, feature_list=list(range(52))):
    mse = np.mean((data - data_pred) ** 2, axis=1)
    #     print(mse.shape)
    mse[mse > threshold] = 1
    mse[mse == threshold] = 0
    mse[mse < threshold] = 0
    return mse


# calculate the detection rate for 20 types of fault
def prediction_error_fault_x_excel(model, scaler, filename=None, excel_name="pred_error_excel_temp.xlsx"):
    # calculate the mean, std, min, max of dataset
    if filename != None:
        data = pd.read_csv(filename, sep=",", header=None, ).to_numpy()[:, 1:]

    frames = []
    list_1 = [1, 2, 4, 6, 7, 8, 11, 12, 13, 14, 17, 18]
    list_2 = [5, 10, 16, 19, 20]
    list_3 = [3, 9, 15]

    feature_name = ["fault_index", "xmeas_1", "xmeas_2", "xmeas_3", "xmeas_4", "xmeas_5", "xmeas_6", "xmeas_7",
                    "xmeas_8", "xmeas_9", "xmeas_10", "xmeas_11", "xmeas_12", "xmeas_13", "xmeas_14", "xmeas_15",
                    "xmeas_16", "xmeas_17", "xmeas_18", "xmeas_19", "xmeas_20", "xmeas_21", "xmeas_22", "xmeas_23",
                    "xmeas_24", "xmeas_25", "xmeas_26", "xmeas_27", "xmeas_28", "xmeas_29", "xmeas_30", "xmeas_31",
                    "xmeas_32", "xmeas_33", "xmeas_34", "xmeas_35", "xmeas_36", "xmeas_37", "xmeas_38", "xmeas_39",
                    "xmeas_40", "xmeas_41", "xmv_1", "xmv_2", "xmv_3", "xmv_4", "xmv_5", "xmv_6", "xmv_7", "xmv_8",
                    "xmv_9", "xmv_10", "xmv_11"]
    frames.append(pd.DataFrame(feature_name))
    for i in list_1 + list_2 + list_3:
        filename = root_folder + "data_set_csv/faulty_test_fault_x_10%/faulty_test_fault_" + str(i) + ".csv"
        faulty_test_fault_x = pd.read_csv(filename, sep=",", header=None).to_numpy()
        faulty_test_fault_x_sta = scaler.transform(faulty_test_fault_x[:, 1:]) / eta
        faulty_test_fault_x_sta_pred = model.predict(faulty_test_fault_x_sta)
        pred_error = np.absolute(faulty_test_fault_x_sta_pred - faulty_test_fault_x_sta)
        pred_error_mean = np.mean(pred_error, axis=0)
        pred_error_mean = np.reshape(pred_error_mean, [52, 1])
        pred_error_mean = np.vstack((i, pred_error_mean))

        frames.append(pd.DataFrame(pred_error_mean))

    result = pd.concat(frames, axis=1)
    result.to_excel(excel_name, index=False)


def detection_rate_list_data_excel(model, scaler, eta, threshold, list_data_sta, list_data_sta_pred_name,
                                   feature_list=list(range(52)), excel_name="detection_rate_temp.xlsx"):
    list_detection_rate_name = list_data_sta_pred_name.copy()
    list_detection_rate = []

    list_size = len(list_data_sta)
    for i in range(list_size):
        if i == 5 or i == 9 or i == 13:
            print()
        data_sta_pred = model.predict(list_data_sta[i])
        data_sta_pred_clf = detect_with_reconstruct_error(list_data_sta[i], data_sta_pred, threshold, feature_list)
        detection_rate = prediction_error(data_sta_pred_clf, list_data_sta_pred_name[i] + ": ")
        list_detection_rate.append(detection_rate)

    print()
    list_1 = [1, 2, 4, 6, 7, 8, 11, 12, 13, 14, 17, 18]
    list_2 = [5, 10, 16, 19, 20]
    list_3 = [3, 9, 15]

    for i in list_1 + list_2 + list_3:
        #     for i in list_3:
        if i == 11 or i == 19:
            print()
        filename = root_folder + "data_set_csv/faulty_test_fault_x_100%/faulty_test_fault_" + str(i) + ".csv"

        faulty_test_fault_x = pd.read_csv(filename, sep=",", header=None).to_numpy()
        faulty_test_fault_x_sta = scaler.transform(faulty_test_fault_x[:, 1:]) / eta
        faulty_test_fault_x_pred = model.predict(faulty_test_fault_x_sta)
        faulty_test_fault_x_pred_clf = reconstruct_error_classification(faulty_test_fault_x_sta,
                                                                        faulty_test_fault_x_pred, threshold,
                                                                        feature_list)

        filename_2 = "faulty_test_fault_" + str(i) + "_pred: "
        detection_rate = pred_error(faulty_test_fault_x_pred_clf, filename_2, i)
        temp_name = "faulty_test_fault_" + str(i)
        list_detection_rate_name.append(temp_name)
        list_detection_rate.append(detection_rate)
    #         print(list_detection_rate_name)
    array_detection_rate_name = np.array(list_detection_rate_name)
    shape0 = array_detection_rate_name.shape[0]
    #     print(shape0)
    array_detection_rate_name = np.reshape(array_detection_rate_name, (shape0, 1))

    array_detection_rate = np.array(list_detection_rate)
    shape0 = array_detection_rate.shape[0]
    array_detection_rate = np.reshape(array_detection_rate, (shape0, 1))
    frames = [pd.DataFrame(array_detection_rate_name), pd.DataFrame(array_detection_rate)]
    result = pd.concat(frames, axis=1)
    result.to_excel(excel_name, index=False)


# In[10]:


def print_model_history(history):
    # list all data in history
    print(history.history.keys())

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


# In[11]:


def save_model_weights(model, model_pathAndname, weight_pathAndname):
    # serialize model to JSON
    model_json = model.to_json()

    with open(model_pathAndname, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(weight_pathAndname)
    print("Saved model and weights to disk")


def load_model_weights(model_pathAndname, weight_pathAndname):
    # load json and create model
    json_file = open(model_pathAndname, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weight_pathAndname)
    print("Loaded model and weights from disk")
    return loaded_model


# In[12]:


def plot_pred_2col(
        plot_range, features, fault_free_sta, fault_free_sta_pred, faulty_sta, faulty_sta_pred,
):
    color1 = "y"
    color2 = "b"
    color3 = "r"
    color4 = "g"
    label1 = "fault_free"
    label2 = "fault_free_pred"
    label3 = "faulty"
    label4 = "faulty_pred"

    #     fig, (ax1, ax2, ax3) = plt.subplots(3)
    #     features_size = len(features)

    for feature in features:
        fig, axarr = plt.subplots(3, 2)

        axarr[0, 0].plot(range(plot_range), fault_free_sta[:plot_range, feature],
                         label=label1 + "-fea: " + str(feature), color="y", )

        axarr[1, 0].plot(range(plot_range), fault_free_sta_pred[:plot_range, feature],
                         label=label2 + "-fea: " + str(feature), color="g", )

        axarr[2, 0].plot(range(plot_range), fault_free_sta[:plot_range, feature],
                         label=label1 + "-fea: " + str(feature), color="y", )
        axarr[2, 0].plot(range(plot_range), fault_free_sta_pred[:plot_range, feature],
                         label=label2 + "-fea: " + str(feature),
                         color="g", )

        axarr[0, 1].plot(range(plot_range), faulty_sta[:plot_range, feature], label=label3 + "-fea: " + str(feature),
                         color="y", )
        axarr[1, 1].plot(range(plot_range), faulty_sta_pred[:plot_range, feature],
                         label=label4 + "-fea: " + str(feature), color="r", )
        axarr[2, 1].plot(range(plot_range), faulty_sta[:plot_range, feature], label=label3 + "-fea: " + str(feature),
                         color="y", )
        axarr[2, 1].plot(range(plot_range), faulty_sta_pred[:plot_range, feature],
                         label=label4 + "-fea: " + str(feature), color="r", )

        axarr[0, 0].set_title("feature: " + str(feature) + ", " + "fault_free_sta")
        axarr[0, 1].set_title("faulty_sta")

    plt.show()
    plt.close("all")
    fig.clf()


# In[13]:


def plot_pred_4col(plot_range, features, fault_free_train_sta, fault_free_train_sta_pred, fault_free_test_sta,
                   fault_free_test_sta_pred, faulty_train_sta, faulty_train_sta_pred, faulty_test_sta,
                   faulty_test_sta_pred, ):
    color1 = "y"
    color2 = "b"
    color3 = "r"
    color4 = "g"
    label1 = "fault_free"
    label2 = "fault_free_pred"
    label3 = "faulty"
    label4 = "faulty_pred"

    #     fig, (ax1, ax2, ax3) = plt.subplots(3)
    #     features_size = len(features)

    for feature in features:
        fig, axarr = plt.subplots(3, 4)

        axarr[0, 0].plot(range(plot_range), fault_free_train_sta[:plot_range, feature], color="y", )

        axarr[1, 0].plot(range(plot_range), fault_free_train_sta_pred[:plot_range, feature], color="g", )

        axarr[2, 0].plot(range(plot_range), fault_free_train_sta[:plot_range, feature], color="y", )
        axarr[2, 0].plot(range(plot_range), fault_free_train_sta_pred[:plot_range, feature], color="g", )

        axarr[0, 1].plot(range(plot_range), fault_free_test_sta[:plot_range, feature], color="y", )
        axarr[1, 1].plot(range(plot_range), fault_free_test_sta_pred[:plot_range, feature], color="r", )
        axarr[2, 1].plot(range(plot_range), fault_free_test_sta[:plot_range, feature], color="y", )
        axarr[2, 1].plot(range(plot_range), fault_free_test_sta_pred[:plot_range, feature], color="r", )

        axarr[0, 2].plot(range(plot_range), faulty_train_sta[:plot_range, feature], color="y", )

        axarr[1, 2].plot(range(plot_range), faulty_train_sta_pred[:plot_range, feature], color="g", )

        axarr[2, 2].plot(range(plot_range), faulty_train_sta[:plot_range, feature], color="y", )
        axarr[2, 2].plot(range(plot_range), faulty_train_sta_pred[:plot_range, feature], color="g", )

        axarr[0, 3].plot(range(plot_range), faulty_test_sta[:plot_range, feature], color="y", )
        axarr[1, 3].plot(range(plot_range), faulty_test_sta_pred[:plot_range, feature], color="r", )
        axarr[2, 3].plot(range(plot_range), faulty_test_sta[:plot_range, feature], color="y", )
        axarr[2, 3].plot(range(plot_range), faulty_test_sta_pred[:plot_range, feature], color="r", )

        axarr[0, 0].set_title("feature: " + str(feature) + ", " + "fault_free_train_sta")
        axarr[0, 1].set_title("fault_free_test_sta")
        axarr[0, 2].set_title("faulty_train_sta")
        axarr[0, 3].set_title("faulty_test_sta")

    plt.show()
    plt.close("all")
    fig.clf()


def plot_pred_overlap(plot_range, features, fault_free_sta, fault_free_sta_pred, faulty_sta, faulty_sta_pred, ):
    color1 = "y"
    color2 = "b"
    color3 = "r"
    color4 = "g"
    label1 = "fault_free"
    label2 = "fault_free_pred"
    label3 = "faulty"
    label4 = "faulty_pred"

    #     fig, (ax1, ax2, ax3) = plt.subplots(3)
    #     features_size = len(features)

    for feature in features:
        fig, axarr = plt.subplots(1, 2)

        axarr[0].plot(
            range(plot_range), fault_free_sta[:plot_range, feature], label=label1 + "-fea: " + str(feature),
            color=color1,
        )
        axarr[0].plot(
            range(plot_range), fault_free_sta_pred[:plot_range, feature], label=label2 + "-fea: " + str(feature),
            color="g",
        )
        #         axarr[0].legend()

        axarr[1].plot(
            range(plot_range), faulty_sta[:plot_range, feature], label=label3 + "-fea: " + str(feature), color=color1,
        )
        axarr[1].plot(
            range(plot_range), faulty_sta_pred[:plot_range, feature], label=label4 + "-fea: " + str(feature), color="r",
        )
        #         axarr[1].legend()

        axarr[0].set_title("feature: " + str(feature) + ", " + "fault_free_train_sta")
        axarr[1].set_title("fault_free_test_sta")
    plt.show()
    plt.close("all")
    fig.clf()


def plot_pred_fault_x(model, fault_index, scaler, features=list(range(52)), plot_range=5000):
    list_1 = [1, 2, 4, 6, 7, 8, 11, 12, 13, 14, 17, 18]
    list_2 = [5, 10, 16, 19, 20]
    list_3 = [3, 9, 15]

    i = fault_index
    filename = root_folder + "data_set_csv/faulty_test_fault_x_10%/faulty_test_fault_" + str(i) + ".csv"
    faulty_test_fault_x = pd.read_csv(filename, sep=",", header=None).to_numpy()
    faulty_test_fault_x_sta = scaler.transform(faulty_test_fault_x[:, 1:]) / eta
    faulty_test_fault_x_sta_pred = model.predict(faulty_test_fault_x_sta)
    plot_pred_overlap(
        plot_range, features, fault_free_train_20000_sta, fault_free_train_20000_sta_pred, faulty_test_fault_x_sta,
        faulty_test_fault_x_sta_pred)


def cal_mean_std_min_max_sta(data, scaler, eta=2, filename=None, excel_name="temp_excel.xlsx"):
    # calculate the mean, std, min, max of dataset
    if filename != None:
        data = pd.read_csv(filename, sep=",", header=None, ).to_numpy()[:, 1:]

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    _min = np.min(data, axis=0)
    _max = np.max(data, axis=0)

    _sta_min = np.zeros(52)
    _sta_max = np.zeros(52)
    for i in range(52):
        _sta_min[i] = scaler.transform(data)[:, i].min() / eta
        _sta_max[i] = scaler.transform(data)[:, i].max() / eta

    frames = [
        pd.DataFrame(mean),
        pd.DataFrame(std),
        pd.DataFrame(_min),
        pd.DataFrame(_max),
        pd.DataFrame(_sta_min),
        pd.DataFrame(_sta_max),
    ]
    result = pd.concat(frames, axis=1)
    result.to_excel(excel_name, index=False)

    return mean, std, _min, _max, _sta_min, _sta_max


def AE_train(model, data_list, data_name_list, epochs=500):
    input_train_data, _ = get_data_from_data_list("fault_free_train_100000", data_name_list, data_list)
    output_train_data = input_train_data

    input_validation_data, _ = get_data_from_data_list("fault_free_test_100000", data_name_list, data_list)
    output_validation_data = input_validation_data

    # keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

    epochs = epochs
    model_fit_verbose = 2

    es = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=2, mode="min")

    history = model.fit(
        input_train_data,
        output_train_data,
        epochs=epochs,
        verbose=model_fit_verbose,
        validation_data=(input_validation_data, output_validation_data),
        callbacks=[es],
    )

    print_model_history(history)
    return model


def AE_test(model, data_list, data_name_list, threshold, feature_list=list(range(52))):
    # tightness = 0.98  # fix this for comparasion
    # threshold = 1 - tightness
    # threshold = threshold

    # adjust threshold, keep false alarm around 0.1
    # input_train_data, _ = get_data_from_data_list("fault_free_train_250000", data_name_list, data_list)
    # input_train_data_pred = model.predict(input_train_data)
    # false_alarm = detect_binary_classification(input_train_data_pred, threshold, feature_list)

    # change threshold if not satisfy the 0.1 constrain
    # temp_counter = 0

    # print("threshold: " + str(threshold))
    # print()

    # detect for all data in list_data_sta
    # detection_rate_list_data_excel(model, scaler, eta, threshold, list_data_sta, list_data_sta_pred_name,                                   excel_name=excel_name)
    faulty_train_Xp, _ = get_data_from_data_list("faulty_train_10p", data_name_list, data_list)

    faulty_train_Xp_pred = model.predict(faulty_train_Xp)
    detection_rate = detect_binary_classification(faulty_train_Xp, faulty_train_Xp_pred, threshold)
    return detection_rate


def AE_test_all(model, data_list, data_name_list, scaler, eta, excel_name="detection_rate_excel_temp.xlsx",
                feature_list=list(range(52))):
    _, threshold, _ = adjust_threshold_AE(model, data_list, data_name_list, threshold=0.2,
                                          step_size=0.002,
                                          tolerance=0.003, ref_detection_rate=0.1, feature_list=list(range(52)))
    list_detection_rate_name = data_name_list.copy()
    list_detection_rate = []

    for data_name in data_name_list:
        data, _ = get_data_from_data_list(data_name, data_name_list, data_list)
        data_pred = model.predict(data)
        detection_rate = detect_binary_classification(data, data_pred, threshold, feature_list)
        list_detection_rate.append(detection_rate)

    list_1 = [1, 2, 4, 6, 7, 8, 11, 12, 13, 14, 17, 18]
    list_2 = [5, 10, 16, 19, 20]
    list_3 = [3, 9, 15]
    for i in list_1 + list_2 + list_3:
        #     for i in list_3:
        # if i == 11 or i == 19:
        #     print()
        root_folder = ""
        filepath = root_folder + "data_set_csv/faulty_test_fault_x_10p/faulty_test_fault_" + str(i) + ".csv"
        faulty_test_fault_x = pd.read_csv(filepath, sep=",", header=None).to_numpy()
        faulty_test_fault_x_sta = scaler.transform(faulty_test_fault_x[:, 1:]) / eta
        faulty_test_fault_x_pred = model.predict(faulty_test_fault_x_sta)
        detection_rate = detect_binary_classification(faulty_test_fault_x_sta,
                                                      faulty_test_fault_x_pred, threshold,
                                                      feature_list)
        temp_name = "faulty_test_fault_" + str(i)
        list_detection_rate_name.append(temp_name)
        list_detection_rate.append(detection_rate)
    array_detection_rate_name = np.array(list_detection_rate_name)
    shape0 = array_detection_rate_name.shape[0]
    array_detection_rate_name = np.reshape(array_detection_rate_name, (shape0, 1))
    array_detection_rate = np.array(list_detection_rate)
    shape0 = array_detection_rate.shape[0]
    array_detection_rate = np.reshape(array_detection_rate, (shape0, 1))
    frames = [pd.DataFrame(array_detection_rate_name), pd.DataFrame(array_detection_rate)]
    result = pd.concat(frames, axis=1)
    result.to_excel(excel_name, index=False)


def binary_classifier_test(model, data_list, data_name_list, scaler, eta, excel_name="detection_rate_excel_temp.xlsx",
                           feature_list=list(range(52))):
    list_detection_rate_name = data_name_list.copy()
    list_detection_rate = []

    for data_name in data_name_list:
        data, data_label = get_data_from_data_list(data_name, data_name_list, data_list)
        data_clf = model.predict(data)
        detection_rate = np.count_nonzero(data_clf) / data_clf.shape[0]
        # detection_rate = np.sum(data_label.astype(int) == data_clf.astype(int))/data_label.shape[0]
        list_detection_rate.append(detection_rate)

    list_1 = [1, 2, 4, 6, 7, 8, 11, 12, 13, 14, 17, 18]
    list_2 = [5, 10, 16, 19, 20]
    list_3 = [3, 9, 15]
    for i in list_1 + list_2 + list_3:
        #     for i in list_3:
        # if i == 11 or i == 19:
        #     print()
        root_folder = ""
        filepath = root_folder + "data_set_csv/faulty_test_fault_x_10p/faulty_test_fault_" + str(i) + ".csv"
        faulty_test_fault_x = pd.read_csv(filepath, sep=",", header=None).to_numpy()
        faulty_test_fault_x_sta = scaler.transform(faulty_test_fault_x[:, 1:]) / eta
        faulty_test_fault_x_clf = model.predict(faulty_test_fault_x_sta)
        detection_rate = np.count_nonzero(faulty_test_fault_x_clf) / faulty_test_fault_x_clf.shape[0]
        # detection_rate = np.sum(faulty_test_fault_x[:, 0].astype(int) == faulty_test_fault_x_clf.astype(int))/faulty_test_fault_x.shape[0]

        temp_name = "faulty_test_fault_" + str(i)
        list_detection_rate_name.append(temp_name)
        list_detection_rate.append(detection_rate)
    array_detection_rate_name = np.array(list_detection_rate_name)
    shape0 = array_detection_rate_name.shape[0]
    array_detection_rate_name = np.reshape(array_detection_rate_name, (shape0, 1))
    array_detection_rate = np.array(list_detection_rate)
    shape0 = array_detection_rate.shape[0]
    array_detection_rate = np.reshape(array_detection_rate, (shape0, 1))
    frames = [pd.DataFrame(array_detection_rate_name), pd.DataFrame(array_detection_rate)]
    result = pd.concat(frames, axis=1)
    result.to_excel(excel_name, index=False)


def multi_classifier_test(model, data_list, data_name_list, scaler, eta, excel_name="detection_rate_excel_temp.xlsx",
                          feature_list=list(range(52))):
    list_detection_rate_name = data_name_list.copy()
    list_detection_rate = []

    for data_name in data_name_list:
        data, data_label = get_data_from_data_list(data_name, data_name_list, data_list)
        data_clf = model.predict(data)
        data_clf = np.reshape(data_clf, (data_clf.shape[0], 1))
        # detection_rate =np.count_nonzero(data_clf) / data_clf.shape[0]
        detection_rate = np.sum(data_label.astype(int) == data_clf.astype(int)) / data_label.shape[0]
        list_detection_rate.append(detection_rate)

    list_1 = [1, 2, 4, 6, 7, 8, 11, 12, 13, 14, 17, 18]
    list_2 = [5, 10, 16, 19, 20]
    list_3 = [3, 9, 15]
    for i in list_1 + list_2 + list_3:
        #     for i in list_3:
        # if i == 11 or i == 19:
        #     print()
        root_folder = ""
        filepath = root_folder + "data_set_csv/faulty_test_fault_x_10p/faulty_test_fault_" + str(i) + ".csv"
        faulty_test_fault_x = pd.read_csv(filepath, sep=",", header=None).to_numpy()
        faulty_test_fault_x_sta = scaler.transform(faulty_test_fault_x[:, 1:]) / eta
        faulty_test_fault_x_clf = model.predict(faulty_test_fault_x_sta)
        faulty_test_fault_x_clf = np.reshape(faulty_test_fault_x_clf, (faulty_test_fault_x_clf.shape[0], 1))
        # detection_rate = np.count_nonzero(faulty_test_fault_x_clf) / faulty_test_fault_x_clf.shape[0]
        detection_rate = np.sum(faulty_test_fault_x[:, 0].astype(int) == faulty_test_fault_x_clf.astype(int)) / \
                         faulty_test_fault_x.shape[0]

        temp_name = "faulty_test_fault_" + str(i)
        list_detection_rate_name.append(temp_name)
        list_detection_rate.append(detection_rate)
    array_detection_rate_name = np.array(list_detection_rate_name)
    shape0 = array_detection_rate_name.shape[0]
    array_detection_rate_name = np.reshape(array_detection_rate_name, (shape0, 1))
    array_detection_rate = np.array(list_detection_rate)
    shape0 = array_detection_rate.shape[0]
    array_detection_rate = np.reshape(array_detection_rate, (shape0, 1))
    frames = [pd.DataFrame(array_detection_rate_name), pd.DataFrame(array_detection_rate)]
    result = pd.concat(frames, axis=1)
    result.to_excel(excel_name, index=False)


# def model_test(model, data_list, data_name_list, feature_list=list(range(52))):
#     for data_name in data_name_list:
#         # print(data_name)
#         data_temp = get_data_from_data_list(data_name, data_name_list, data_list)[:, 1:]
#         label_temp = get_data_from_data_list(data_name, data_name_list, data_list)[:, 0]
#         label_pred = model.predict(data_temp)
#         detection_rate = detect_binary_classification(label_pred, data_name)
#         print(data_name + ": " + str(detection_rate))


def adjust_threshold_AE(model, data_list, data_name_list, threshold=0.2,
                        step_size=0.002,
                        tolerance=0.003, ref_detection_rate=0.1, feature_list=list(range(52))):
    fault_free_data, _ = get_data_from_data_list("fault_free_train_250000", data_name_list, data_list)
    fault_free_data_pred = model.predict(fault_free_data)

    threshold_temp = threshold
    step_size_temp = step_size
    detection_rate = detect_binary_classification(fault_free_data, fault_free_data_pred, threshold_temp, feature_list)
    counter = 0
    while np.abs(detection_rate - ref_detection_rate) > tolerance:
        sign_value_1 = np.sign(detection_rate - ref_detection_rate)
        threshold_temp = threshold_temp + sign_value_1 * step_size_temp
        detection_rate = detect_binary_classification(fault_free_data, fault_free_data_pred, threshold_temp,
                                                      feature_list)
        sign_value_2 = np.sign(detection_rate - ref_detection_rate)
        if np.abs(detection_rate - ref_detection_rate) < tolerance:
            return detection_rate, threshold_temp, step_size_temp
        elif sign_value_1 != sign_value_2:
            threshold_temp = threshold_temp - sign_value_1 * step_size_temp
            step_size_temp = step_size_temp / 2
            threshold_temp = threshold_temp + sign_value_2 * step_size_temp
        else:
            threshold_temp = threshold_temp + sign_value_1 * step_size_temp
        detection_rate = detect_binary_classification(fault_free_data, fault_free_data_pred, threshold_temp,
                                                      feature_list)
        counter = counter + 1
        if (counter > 100):
            return -1, -1, -1
    return detection_rate, threshold_temp, step_size_temp


def classifier_ROC(model, testX, testy, label):
    ns_probs = [0 for _ in range(len(testy))]
    lr_probs = model.predict_proba(testX)
    lr_probs = lr_probs[:, 1]
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
    pyplot.plot(lr_fpr, lr_tpr, marker=".", label=label)
    # axis labels
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    auc = roc_auc_score(testy, lr_probs)
    print('AUC: %.3f' % auc)


def classifier_calibration_curve(model, testX, testy, label):
    prob_pos = model.predict_proba(testX)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(testy, prob_pos, n_bins=10)
    fop, mpv = calibration_curve(testy, prob_pos, n_bins=10, normalize=True)
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    # plot model reliability
    pyplot.plot(mpv, fop, marker=".", label=label)
    pyplot.legend()
    pyplot.show()


# create the label for fault free data and faulty data, concatenate them
def concat_binary_label_data(fault_free_data, faulty_data):
    # create labels
    fault_free_label = np.zeros((fault_free_data.shape[0], 1))
    faulty_label = np.ones((faulty_data.shape[0], 1))

    a1 = np.hstack((fault_free_label, fault_free_data))
    a2 = np.hstack((faulty_label, faulty_data))
    concat_data = np.vstack((a1, a2))
    return concat_data[:, 1:], concat_data[:, 0]


def get_data_from_data_list(data_name, data_name_list, data_list):
    data_name_index = data_name_list.index(data_name)

    data = data_list[data_name_index][:, 1:]
    label = data_list[data_name_index][:, 0]
    return data, label
