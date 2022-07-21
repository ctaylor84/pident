import warnings
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy.stats as sps

from util.data_sampling import GroupSampling
from pident_cluster import BENCHMARK_MIN_TRIM as INDICES_MIN_INDEX

DAY_SECONDS = 86400
PIG_COUNTS = [10,20,30,40,50]
SAMPLE_RATES = [1.0,2.0,3.0,4.0]
PREDICTION_PLOT = False

def GetTrueTrajectory(group_mat, animal_id):
    series_i = group_mat[group_mat[:,2] == animal_id]
    if series_i.ndim == 1:
        series_i = np.expand_dims(series_i, 0)
    return series_i[np.argsort(series_i[:,1])]

def GetPredTrajectory(group_mat, pred_indices, animal_index):
    pred_i = pred_indices["pred"][pred_indices["true"] == animal_index]
    series_i = group_mat[pred_indices["clusters"] == pred_i]
    if series_i.ndim == 1:
        series_i = np.expand_dims(series_i, 0)
    return series_i[np.argsort(series_i[:,1])]

def GetTrajectories(group_mat, pred_indices):
    animal_ids = np.unique(group_mat[:,2])
    series_starts = list()
    series_ends = list()
    in_series = defaultdict(list)
    for i in range(animal_ids.shape[0]):
        series_i = GetTrueTrajectory(group_mat, animal_ids[i])
        in_series["true"].append(series_i)
        series_starts.append(series_i[0,1])
        series_ends.append(series_i[-1,1])

    for i in range(animal_ids.shape[0]):
        series_i = GetPredTrajectory(group_mat, pred_indices, i)
        in_series["pred"].append(series_i)
        series_starts.append(series_i[0,1])
        series_ends.append(series_i[-1,1])

    in_series["group"] = group_mat[np.argsort(group_mat[:,1])]
    return in_series

def GetMatchedRawData(format_mode, pig_data, group_mat):
    animal_ids = np.unique(group_mat[:,2])
    series_ends = list()
    raw_series = list()
    for animal_id in animal_ids:
        series_ends.append(np.amax(pig_data[animal_id][:,3]))
        raw_series.append(pig_data[animal_id])

    if format_mode == "true" or format_mode == "pred":
        matched_series = list()
        for i in range(animal_ids.shape[0]):
            series_i = np.stack((raw_series[i][:,5], raw_series[i][:,3], raw_series[i][:,1]), axis=1)
            matched_series.append(series_i)
    elif format_mode == "group":
        group_series = np.concatenate(raw_series)
        group_series = np.stack((group_series[:,5], group_series[:,3], group_series[:,1]), axis=1)
        group_series = group_series[np.argsort(group_series[:,1])]
        matched_series = group_series
    else:
        raise ValueError("GetMatchedRawData: format_mode must equal 'true' or 'pred' or 'group'")
    return matched_series

def ImputationAverage(series, pred_times, avg_growth):
    assert series.shape[0] == 1
    fixed_weight = series[0,0]
    fixed_time = series[0,1]
    pred_weights = [0.0] * len(pred_times)
    for i, pred_time in enumerate(pred_times):
        pred_weights[i] = fixed_weight + (pred_time - fixed_time) * avg_growth
    return pred_weights

def ImputationRegression(series, pred_times, quadratic=False):
    series_x = series[:,1]
    series_y = series[:,0]
    if quadratic:
        series_x = np.stack((series_x, series_x * series_x), axis=1)
        pred_x = np.stack((pred_times, pred_times * pred_times), axis=1)
    else:
        series_x = series_x.reshape(-1, 1)
        pred_x = pred_times.reshape(-1, 1)
    rgs = LinearRegression().fit(series_x, series_y)
    return rgs.predict(pred_x)

def PredictionPlot(x_pred, x_step, y_raw, y_step, current_step):
    plt.figure(figsize=(12,4))
    seq_length = round(y_raw[-1,1] // DAY_SECONDS)
    time_seq = np.arange(seq_length + 1) + 0.5
    x_step_c = np.asarray([x / 1000 for x in x_step])

    y_raw_time_min = (current_step+7+14-1) * DAY_SECONDS
    y_raw_c = y_raw[y_raw[:,1] > y_raw_time_min]
    y_raw_c = y_raw_c[y_raw_c[:,1] < y_raw_time_min + DAY_SECONDS]
    # plt.scatter(y_raw_c[:,1] / DAY_SECONDS, y_raw_c[:,0] / 1000, c="c", marker="+", s=30, label="Future RFID weights")
    
    plt.scatter(x_pred[:,1] / DAY_SECONDS, x_pred[:,0] / 1000, c="r", marker="+", s=30, label="Growth trajectory")
    x_digitized = np.digitize(x_pred[:,1] / DAY_SECONDS, time_seq[current_step:current_step+8] - 0.5)
    x_pred_disc = np.asarray([x_pred[x_digitized == i,0].shape[0] for i in range(1, 8)])
    time_seq_f = time_seq[current_step:current_step+7]
    if x_pred_disc.any() > 0:
        plt.scatter(time_seq_f[x_pred_disc > 0], x_step_c[x_pred_disc > 0], c="b", marker="x", s=30, label="Input weights")
    if (x_pred_disc == 0).any():
        plt.scatter(time_seq_f[x_pred_disc == 0], x_step_c[x_pred_disc == 0], c="b", marker="D", s=20, label="Imputed input weights")
    plt.scatter(time_seq[current_step+7+14-1], y_step[current_step+7+14-1] / 1000, c="g", marker="o", s=30, label="Ground-truth / future weight")

    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ylim_min, ylim_max = ax.get_ylim()
    plt.vlines(time_seq[current_step:current_step+8] - 0.5, ylim_min, ylim_max, label="Input weight bin limits")
    plt.vlines(time_seq[current_step+7+14-1] - 0.5, ylim_min, ylim_max, linestyles="dashed", label="Future weight bin limits")
    plt.vlines(time_seq[current_step+7+14-1] + 0.5, ylim_min, ylim_max, linestyles="dashed")
    plt.legend(loc=0, title="Legend", frameon=False)
    plt.xticks(np.arange(29))
    plt.show()

def DiscretizeSeries(series, step_index, step_size, frag_size, avg_growth, std_calc=False):
    series = series[np.argsort(series[:,1])]
    step_seconds = round(DAY_SECONDS * step_size)
    if step_index == None:
        seq_length = round(series[-1,1] // step_seconds)
        time_seq = np.arange(0.0, (seq_length + 1) * step_seconds, step_seconds)
    else:    
        if step_index < 0:
            raise ValueError("Step index is less than 0")
        seq_length = frag_size
        seq_start = step_index * step_seconds
        time_seq = np.arange(seq_start, seq_start + (seq_length + 1) * step_seconds, step_seconds)
    cut_series = series[series[:,1] < time_seq[-1]]
    digitized = np.digitize(cut_series[:,1], time_seq)

    weights = [cut_series[digitized == i,0] for i in range(1, time_seq.shape[0])]
    weight_means = [0.0] * seq_length
    weight_stds = [0.0] * seq_length
    weight_exist = [False] * seq_length
    for i in range(seq_length):
        weight_count = weights[i].shape[0]
        if weight_count > 0:
            weight_exist[i] = True
            weight_means[i] = np.mean(weights[i])
            if std_calc:
                weight_stds[i] = np.std(weights[i])

    if sum(weight_exist) != seq_length:
        if step_index != None:
            impute_times = time_seq + step_seconds / 2
            if cut_series.shape[0] == 0:
                raise RuntimeError("Insufficient data for imputation")
            elif cut_series.shape[0] <= 4:
                impute_pred = ImputationAverage(np.mean(cut_series, axis=0, keepdims=True), 
                                                impute_times, avg_growth)
            else:
                impute_pred = ImputationRegression(cut_series, impute_times)

            for i in range(seq_length):
                if not weight_exist[i]:
                        weight_means[i] = impute_pred[i]
        else:
            for i in range(seq_length):
                if not weight_exist[i]:
                    weight_means[i] = None

    if not std_calc:
        return weight_means
    else:
        return weight_means, weight_stds

def ForecastFormat(format_mode, x_series, y_series_indv, y_series_group, pred_indices, 
                   avg_growth, group_mat, min_pigs_ratio, step_size, frag_size, 
                   forecast_horizon, group_limits, series_length):
    if frag_size < INDICES_MIN_INDEX:
        raise ValueError("frag_size is smaller than INDICES_MIN_INDEX")
    series_indices = list(range(len(x_series["true"])))
    forecast_x = list()
    forecast_y = list()
    forecast_steps = list()
    forecast_counts = defaultdict(int)
    min_pigs = round(min_pigs_ratio * len(x_series["true"]))

    series_iter = zip(series_indices, x_series["true"], y_series_indv)
    for series_index, x_series_true_d, y_series_i in series_iter:
        y_series_d = DiscretizeSeries(y_series_i, None, step_size, frag_size, avg_growth)
        step_count = min(len(y_series_d)-frag_size-forecast_horizon+1, series_length-frag_size+1)
        if step_count < 1:
            raise ValueError("Insufficient data for series" + str(series_index))
        for i in range(step_count):
            if group_limits[i + frag_size - INDICES_MIN_INDEX]:
                continue
            if y_series_d[i+frag_size+forecast_horizon-1] == None:
                continue

            if format_mode == "pred":
                pred_indices_i = pred_indices[i + frag_size - INDICES_MIN_INDEX]
                trim_index = pred_indices_i["clusters"].shape[0]
                group_mat_i = group_mat[:trim_index]
                x_series_pred = GetPredTrajectory(group_mat_i, pred_indices_i, series_index)
                x_series_step = DiscretizeSeries(x_series_pred, i, step_size, frag_size, avg_growth)
            elif format_mode == "true":
                x_series_step = DiscretizeSeries(x_series_true_d, i, step_size, frag_size, avg_growth)

            if format_mode != "group":
                forecast_x.append(np.array(x_series_step))
                forecast_y.append(y_series_d[i+frag_size+forecast_horizon-1])

            if PREDICTION_PLOT and format_mode == "pred" and series_index == 0 and i == 7:
                # Series index: 0 Step index: 7
                print("Series index:", series_index, "Step index:", i)
                PredictionPlot(x_series_pred, x_series_step, y_series_i, y_series_d, i)
            forecast_steps.append(i)
            forecast_counts[i] += 1

    if format_mode == "group":
        forecast_steps = list()
        y_series_g_mean, y_series_g_std = DiscretizeSeries(y_series_group, None, step_size, 
                                                           frag_size, avg_growth, std_calc=True)
        for step_index, pig_count in forecast_counts.items():
            if pig_count >= min_pigs:
                step_x_mean, step_x_std = DiscretizeSeries(x_series["group"], step_index, 
                                            step_size, frag_size, avg_growth, std_calc=True)
                step_x = np.array(step_x_mean + step_x_std + [pig_count])
                forecast_x.append(step_x)
                forecast_y.append(np.asarray([y_series_g_mean[step_index+frag_size+forecast_horizon-1], 
                                              y_series_g_std[step_index+frag_size+forecast_horizon-1]]))
                forecast_steps.append(step_index)
    else:
        for i in reversed(range(len(forecast_steps))):
            pig_count = forecast_counts[forecast_steps[i]]
            if pig_count < min_pigs:
                del forecast_x[i]
                del forecast_y[i]
                del forecast_steps[i]
    return forecast_x, forecast_y, forecast_steps

def GetAvgGrowth(pig_data, fold_sets):
    avg_growth = list()
    for fold_n in range(len(fold_sets)):
        fold_coefs = list()
        train_set = list()
        for i in range(len(fold_sets)):
            if i != fold_n:
                train_set += fold_sets[i]
        for animal_id in train_set:
            pig_data_i = pig_data[animal_id][np.argsort(pig_data[animal_id][:,3])]
            weight_data = pig_data_i[:,5].reshape(-1, 1)
            time_data = pig_data_i[:,3].reshape(-1, 1)
            rgs = LinearRegression().fit(time_data, weight_data)
            fold_coefs.append(rgs.coef_[0])
        avg_growth.append(np.mean(fold_coefs))
    return avg_growth

def GetSeriesLimits(fold_n, pig_count, tj_model, series_count, series_length):
    series_limits = [None] * series_count
    for sample_rate in SAMPLE_RATES:
        file_code = str(pig_count) + "-" + str(sample_rate).replace(".",",") + "-" + str(series_length)
        tj_dir = "model_benchmarks/" + tj_model + "_indices/"
        tj_file_prefix = tj_dir + tj_model + "_benchmark_" + file_code + "_"
        with open(tj_file_prefix + str(fold_n+1) + "_indices.dat", "rb") as fh:
            tj_indices = pickle.load(fh)
        for group_id in range(len(tj_indices)):
            group_indices = tj_indices[group_id]
            if series_limits[group_id] == None:
                series_limits[group_id] = [False] * len(group_indices)
            for group_trim in range(len(group_indices)):
                if group_indices[group_trim] == None:
                    series_limits[group_id][group_trim] = True
    return series_limits

def GenerateDataset(format_mode, tj_model, series_count=100, pig_count=10, sample_rate=1.0, 
                    series_length=35, fold_count=3, min_pigs=0.5, ff_step_size=1.0, 
                    ff_frag_size=7, ff_horizon=14):
    with open("data/data_dump.dat", "rb") as fh:
        pig_data = pickle.load(fh)["pig_series"]
    with open("data/data_folds.dat", "rb") as fh:
        fold_sets = pickle.load(fh)[fold_count]
    avg_growth = GetAvgGrowth(pig_data, fold_sets)

    file_code = str(pig_count) + "-" + str(sample_rate).replace(".",",") + "-" + str(series_length)
    tj_dir = "model_benchmarks/" + tj_model + "_indices/"
    tj_file_prefix = tj_dir + tj_model + "_benchmark_" + file_code + "_"

    dataset_x = list()
    dataset_y = list()
    dataset_folds = list()
    dataset_groups = list()
    dataset_steps = list()
    for fold_n in range(fold_count):
        with open(tj_file_prefix + str(fold_n+1) + "_indices.dat", "rb") as fh:
            tj_indices = pickle.load(fh)
        series_limits = GetSeriesLimits(fold_n, pig_count, tj_model, series_count, series_length)
        for group_id in range(series_count):
            group_mat = GroupSampling(pig_data, group_id=group_id, pig_count=pig_count,
                            sample_rate=sample_rate, fold_set=fold_sets[fold_n],
                            series_length=series_length)
        
            pred_indices = tj_indices[group_id]
            x_series = GetTrajectories(group_mat, pred_indices[-1])
            y_series_indv = GetMatchedRawData("true", pig_data, group_mat)
            y_series_group = GetMatchedRawData("group", pig_data, group_mat)

            group_limits = series_limits[group_id]
            group_x, group_y, step_counts = ForecastFormat(format_mode, x_series, 
                                                y_series_indv, y_series_group, pred_indices, 
                                                avg_growth[fold_n], group_mat, min_pigs, ff_step_size, 
                                                ff_frag_size, ff_horizon, group_limits, series_length)
            dataset_x += group_x
            dataset_y += group_y
            dataset_folds += [fold_n] * len(group_x)
            dataset_groups += [group_id] * len(group_x)
            dataset_steps += step_counts
    dataset_x = np.stack(dataset_x)
    if format_mode != "group":
        dataset_y = np.array(dataset_y)
    else:
        dataset_y = np.stack(dataset_y)
    dataset_folds = np.stack(dataset_folds)
    dataset_groups = np.stack(dataset_groups)
    dataset_steps = np.stack(dataset_steps)
    dataset = {"x":dataset_x, "y":dataset_y, "folds":dataset_folds, "groups":dataset_groups,
               "steps":dataset_steps}
    
    _, fold_counts = np.unique(dataset_folds, return_counts=True)
    print("Dataset size:", dataset_x.shape[0])
    print("Fold sizes:", fold_counts)
    return dataset

def GetWeightDistribution(group_mat_in, pig_count):
    group_mat = group_mat_in[np.argsort(group_mat_in[:,1])]
    group_mat[:,0] = group_mat[:,0] / 1000
    time_seq = np.arange(group_mat[0,1], group_mat[-1,1], DAY_SECONDS)
    digitized = np.digitize(group_mat[:,1], time_seq)
    weights = [group_mat[digitized == i,0] for i in range(1, time_seq.shape[0]+1)]
    animal_ids = [group_mat[digitized == i,2] for i in range(1, time_seq.shape[0]+1)]

    rgs = LinearRegression().fit(group_mat[:,1].reshape(-1, 1), group_mat[:,0].reshape(-1, 1))
    pred_series = rgs.predict((time_seq + DAY_SECONDS / 2).reshape(-1, 1))

    min_pigs = round(pig_count * 0.75)
    weight_distributions = list()
    for i in range(time_seq.shape[0]):
        weights_i = np.asarray(weights[i])
        animal_ids_i = np.asarray(animal_ids[i])
        weight_means = list()
        for animal_id in np.unique(animal_ids_i):
            animal_weights = weights_i[animal_ids_i == animal_id]
            if animal_weights.shape[0] == 1:
                weight_means.append(weights_i[0])
            else:
                weight_means.append(np.mean(animal_weights))
        if len(weight_means) > 0: # min_pigs
            weight_means = np.asarray(weight_means, dtype=float)
            weight_means -= pred_series[i]
            weight_distributions.append(weight_means)
    return weight_distributions

def GetRfidMat(pig_data, animal_ids):
    rfid_mat = list()
    for animal_id in animal_ids:
        mat_local = np.stack((pig_data[animal_id][:,5], 
                              pig_data[animal_id][:,3],
                              pig_data[animal_id][:,1]), axis=1)
        rfid_mat.append(mat_local)
    rfid_mat = np.concatenate(rfid_mat, axis=0)
    return rfid_mat[np.argsort(rfid_mat[:,1])]

def SeriesNormalityTest(use_rfid=True, fold_count=3, series_count=100, series_length=35):
    warnings.filterwarnings("ignore")
    with open("data/data_dump.dat", "rb") as fh:
        pig_data = pickle.load(fh)["pig_series"]
    with open("data/data_folds.dat", "rb") as fh:
        fold_sets = pickle.load(fh)[fold_count]
    
    if use_rfid:
        fig, ax = plt.subplots(figsize=(12,8), nrows=2, ncols=3)
        sample_rate_iter = [4.0]
    else:
        fig, ax = plt.subplots(figsize=(12,8), nrows=len(SAMPLE_RATES), ncols=len(PIG_COUNTS))
        sample_rate_iter = list(reversed(SAMPLE_RATES))
    ax_l = ax.reshape(-1)
    weight_dists_all = list()
    ax_index = 0
    for pig_count in PIG_COUNTS:
        for sample_rate in sample_rate_iter:
            weight_dists_local = list()
            for fold_n in range(fold_count):
                for group_id in range(series_count):
                    group_mat_s = GroupSampling(pig_data, group_id=group_id, pig_count=pig_count,
                                    sample_rate=sample_rate, fold_set=fold_sets[fold_n],
                                    series_length=series_length)
                    if use_rfid:
                        group_mat = GetRfidMat(pig_data, np.unique(group_mat_s[:,2]))
                    else:
                        group_mat = group_mat_s
                    weight_dists_i = GetWeightDistribution(group_mat, pig_count)
                    weight_dists_all += weight_dists_i
                    weight_dists_local += weight_dists_i
            weight_dists_local = np.concatenate(weight_dists_local)
            shapiro_local = sps.shapiro(weight_dists_local)
            kstest_local = sps.kstest(weight_dists_local, "norm")
            print("PC:", pig_count, "SR:", sample_rate, 
                  "S#:", weight_dists_local.shape[0],
                  "shapiro:", round(shapiro_local[0], 3),
                  "shapiro p:", round(shapiro_local[1], 3),
                  "kstest:", round(kstest_local[0], 3),
                  "kstest p:", round(kstest_local[1], 3))
            sns.histplot(weight_dists_local, bins=30, kde=True, stat="proportion", ax=ax_l[ax_index])
            if use_rfid:
                ax_l[ax_index].title.set_text("Pig count: " + str(pig_count))
            ax_l[ax_index].text(1,0.93, "N = " + str(weight_dists_local.shape[0]), horizontalalignment="right", transform=ax_l[ax_index].transAxes)
            ax_l[ax_index].text(1,0.86, "Shapiro P = {0:03f}".format(shapiro_local[1]), horizontalalignment="right", transform=ax_l[ax_index].transAxes)
            ax_l[ax_index].text(1,0.79, "KS P = {0:03f}".format(kstest_local[1]), horizontalalignment="right", transform=ax_l[ax_index].transAxes)
            ax_l[ax_index].set(xlabel="Weight (Kg)")
            ax_index += 1
    weight_dists_all = np.concatenate(weight_dists_all)
    shapiro_all = sps.shapiro(weight_dists_local)
    kstest_all = sps.kstest(weight_dists_local, "norm")
    print("Total S#:", weight_dists_all.shape[0],
          "shapiro:", round(shapiro_all[0], 3),
          "shapiro p:", round(shapiro_all[1], 3),
          "kstest:", round(kstest_all[0], 3),
          "kstest p:", round(kstest_all[1], 3))
    print("Total sample count:", weight_dists_all.shape[0])
    # sns_plot = sns.histplot(weight_dists_all, bins=30)
    # sns_plot.set_xlabel("Pig weight (Kg)")
    if use_rfid:
        ax_l[-1].set_axis_off()
        plt.tight_layout()
    else:
        fig.text(0.5, 0.06, "Pig count", ha="center", va="center", fontsize="medium")
        fig.text(0.08, 0.5, "Sample rate", ha="center", va="center", rotation="vertical", fontsize="medium")
    plt.show()

if __name__ == "__main__":
    SeriesNormalityTest()
