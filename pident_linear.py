import os
import pickle
import itertools
import numpy as np
from sklearn.linear_model import LinearRegression

from util.data_sampling import GroupSampling
import util.plot_util as pd_plt

INTERP_FREQ = 3600 # number of seconds
DAY_SECONDS = 86400

N_FOLDS = 3
BENCHMARK_SERIES = 100 # per fold
BENCHMARK_PIG_COUNTS = [10,20,30,40,50]
BENCHMARK_SAMPLE_RATES = [1.0,2.0,3.0,4.0]
BENCHMARK_SERIES_LENGTHS = [35]

class LinearGrowthModel:
    def __init__(self, n_folds=3):
        self.n_folds = n_folds
        self.coef = None

    def train(self, data, fold_sets):
        coefficients = list()
        for n_fold in range(self.n_folds):
            train_set = list()
            for n_fold_j in range(self.n_folds):
                if n_fold_j != n_fold:
                    train_set += fold_sets[n_fold_j]
            growth_rates = list()
            for animal_id in train_set:
                times = data[animal_id][:,3]
                weights = data[animal_id][:,5]
                regr = LinearRegression()
                regr.fit(times.reshape(-1, 1), weights.reshape(-1, 1))
                growth_rates.append(regr.coef_[0][0])
            coefficients.append(np.mean(growth_rates))
        self.coef = coefficients
        print("Model coefficients:", self.coef)

    def eval(self, data, n_fold, group_mat, plot=False):
        if self.coef == None:
            raise RuntimeError("Model has not yet been trained")
    
        animal_ids = list(np.unique(group_mat[:,2]))
        true_series = list()
        pred_series = list()
        scores = list()
        for i in range(len(animal_ids)):
            true_series_i = group_mat[group_mat[:,2] == animal_ids[i]]
            if true_series_i.ndim == 1:
                true_series_i = np.expand_dims(true_series_i, 0)
            true_series_i = true_series_i[np.argsort(true_series_i[:,1])]

            rfid_series = np.stack((data[animal_ids[i]][:,5], data[animal_ids[i]][:,3]), axis=1)

            pred_times = np.arange(true_series_i[0,1], true_series_i[-1,1], INTERP_FREQ)
            pred_weights = rfid_series[0,0] + (pred_times - rfid_series[0,1]) * self.coef[n_fold]
            pred_series_i = np.stack((pred_weights, pred_times), axis=1)

            true_interp = np.interp(pred_times, true_series_i[:,1], true_series_i[:,0])
            distances = pred_weights - true_interp
            series_score = np.sqrt(np.mean(distances * distances))

            scores.append(series_score)
            true_series.append(true_series_i)
            pred_series.append(pred_series_i)

        if plot:
            id_range = list(range(len(animal_ids)))
            true_to_pred = dict(zip(id_range, id_range))
            pd_plt.PlotComparisonCombo(true_to_pred, true_series, pred_series)
        return np.mean(scores) / 1000

def ModelBenchmark(data, fold_sets):
    var_ranges = [BENCHMARK_PIG_COUNTS, BENCHMARK_SAMPLE_RATES, BENCHMARK_SERIES_LENGTHS]
    param_combinations = list(itertools.product(*var_ranges))

    linear_growth_model = LinearGrowthModel(n_folds=N_FOLDS)
    linear_growth_model.train(data, fold_sets)

    out_dir = "model_benchmarks/linear"
    os.makedirs(out_dir, exist_ok=True)

    for n_fold in range(N_FOLDS):
        for pig_count, sample_rate, series_length in param_combinations:
            file_code = str(pig_count) + "-" + str(sample_rate).replace(".",",") + "-" + str(series_length)
            out_name = "linear_benchmark_" + file_code + "_" + str(n_fold+1)
            scores = list()
            for group_id in range(BENCHMARK_SERIES):
                    group_mat = GroupSampling(data, group_id=group_id, pig_count=pig_count,
                                              sample_rate=sample_rate, fold_set=fold_sets[n_fold],
                                              series_length=series_length)
                    group_score = linear_growth_model.eval(data, n_fold, group_mat)
                    scores.append(group_score)
            scores = np.asarray(scores)
            with open(out_dir + "/" + out_name + ".dat", "wb") as fh:
                pickle.dump(scores, fh)

def Main():
    with open("data/data_dump.dat", "rb") as fh:
        data = pickle.load(fh)["pig_series"]
    with open("data/data_folds.dat", "rb") as fh:
        fold_sets = pickle.load(fh)[N_FOLDS]
    ModelBenchmark(data, fold_sets)

if __name__ == "__main__":
    Main()