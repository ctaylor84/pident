import os
import sys
import pickle
import argparse
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment

from util.data_sampling import GroupSampling
from util.data_sampling import GetGroupPairs
import util.plot_util as pd_plt
from pident_pr_est import PrEstimate

INTERP_FREQ = 3600 # number of seconds
DAY_SECONDS = 86400

BENCHMARK_SERIES = 100 # per fold
BENCHMARK_PIG_COUNTS = [10,20,30,40,50]
BENCHMARK_SAMPLE_RATES = [1.0,2.0,3.0,4.0]
BENCHMARK_SERIES_LENGTHS = [35]
BENCHMARK_MIN_TRIM = 5
BENCHMARK_VAR_RANGES = [BENCHMARK_PIG_COUNTS, BENCHMARK_SAMPLE_RATES, BENCHMARK_SERIES_LENGTHS]
BENCHMARK_VAR_NAMES = ["Pig count", "Sample rate", "Series length"]
BENCHMARK_COMPARISON = ["gb-ncv", "mlp-ncv", "rf-ncv", "svm_nys-ncv"]

def SeqGen(group_id=0, pig_count=50, sample_rate=1.0, series_length=35, 
           data_cache=None, fold_set=None):
    if data_cache == None:
        with open("data/data_dump.dat", "rb") as fh:
            data_cache = pickle.load(fh)["pig_series"]
    group_mat = GroupSampling(data_cache, group_id=group_id, pig_count=pig_count,
                              sample_rate=sample_rate, fold_set=fold_set,
                              series_length=series_length)
    return group_mat

def PrCalc(group_mat, estimator="svm_rbf", pair_type="full", n_fold=-1, cv_folds=3, n_threads=10):
    pairs = GetGroupPairs(group_mat)
    scores = PrEstimate(pairs, model_name=estimator, pair_type=pair_type, 
                        n_fold=n_fold, n_threads=n_threads, cv_folds=cv_folds)
    indices_1, indices_2 = pairs[:,4].astype(int), pairs[:,5].astype(int)

    dist_mat = np.zeros((group_mat.shape[0], group_mat.shape[0]), dtype=float)
    for i in range(scores.shape[0]):
        dist_mat[indices_1[i], indices_2[i]] = scores[i]
        dist_mat[indices_2[i], indices_1[i]] = scores[i]

    dist_mat -= np.max(dist_mat)
    dist_mat *= -1
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat

def HCluster(dist_mat, n_clusters=50, linkage="complete"):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", 
                                         linkage=linkage)
    clustering.fit(dist_mat)
    return clustering

def GetPredSeries(group_mat, pred_labels, pig_count=50):
    animal_ids = np.unique(group_mat[:,2])

    true_series = list()
    pred_series = list()
    for i in range(pig_count):
        pred_series_i = group_mat[pred_labels == i,:2]
        if pred_series_i.ndim == 1:
            pred_series_i = np.expand_dims(pred_series_i, 0)
        pred_series_i = pred_series_i[np.argsort(pred_series_i[:,1])]
        pred_series.append(pred_series_i)

        true_series_i = group_mat[group_mat[:,2] == animal_ids[i]]
        if true_series_i.ndim == 1:
            true_series_i = np.expand_dims(true_series_i, 0)
        true_series_i = true_series_i[np.argsort(true_series_i[:,1])]
        true_series.append(true_series_i)
    return true_series, pred_series

def AssignmentMatch(group_mat, pred_labels, pig_count, plot_mode=True):
    true_series, pred_series = GetPredSeries(group_mat, pred_labels, pig_count)

    diff_mat = np.zeros((pig_count, pig_count), dtype=float)
    for j in range(pig_count):
        for i in range(pig_count):
            if pred_series[j].shape[0] == 1 and true_series[i].shape[0] != 1:
                if pred_series[j][0,1] >= true_series[i][0,1] and pred_series[j][0,1] <= true_series[i][-1,1]:
                    true_interp = np.interp(pred_series[j][0,1], true_series[i][:,1], true_series[i][:,0])
                    distance = pred_series[i][0,1] - true_interp
                    diff_mat[i,j] = np.sqrt(distance * distance)
                else:
                    diff_mat[i,j] = np.Inf
            elif true_series[i].shape[0] == 1 and pred_series[j].shape[0] != 1:
                if true_series[i][0,1] >= pred_series[j][0,1] and true_series[i][0,1] <= pred_series[j][-1,1]:
                    pred_interp = np.interp(true_series[i][0,1], pred_series[j][:,1], pred_series[j][:,0])
                    distance = pred_series[i][0,1] - pred_interp
                    diff_mat[i,j] = np.sqrt(distance * distance)
                else:
                    diff_mat[i,j] = np.Inf
            elif true_series[i].shape[0] == 1 and pred_series[j].shape[0] == 1:
                if true_series[i][0,1] == pred_series[j][0,1]:
                    distance = pred_series[j][0,0] - true_series[i][0,0]
                    diff_mat[i,j] = np.sqrt(distance * distance)
                else:
                    diff_mat[i,j] = np.Inf
            else:
                overlap_start = max(pred_series[j][0,1], true_series[i][0,1])
                overlap_end = min(pred_series[j][-1,1], true_series[i][-1,1])
                interp_times = np.arange(overlap_start, overlap_end, INTERP_FREQ)
                if interp_times.shape[0] == 0:
                    diff_mat[i,j] = np.Inf
                else:
                    pred_interp = np.interp(interp_times, pred_series[j][:,1], pred_series[j][:,0])
                    true_interp = np.interp(interp_times, true_series[i][:,1], true_series[i][:,0])
                    distances = pred_interp - true_interp
                    diff_mat[i,j] = np.sqrt(np.mean(distances * distances))

    miss_count = 0
    try:
        true_ind, pred_ind = linear_sum_assignment(diff_mat)
    except ValueError:
        miss_count = 1

    if miss_count > 0:
        inf_counts = np.sum(diff_mat == np.Inf, axis=1)
        inf_indices = np.flip(np.argsort(inf_counts))
        while miss_count < len(true_series):
            diff_mat_r = np.delete(diff_mat, inf_indices[:miss_count], axis=0)
            try:
                true_ind, pred_ind = linear_sum_assignment(diff_mat_r)
                break
            except ValueError:
                miss_count += 1
        for i in range(miss_count):
            true_ind[inf_indices[i]:] += 1
    
    final_score = np.mean(diff_mat[true_ind, pred_ind]) / 1000
    true_to_pred = dict(zip(true_ind, pred_ind))
    pred_indices = {"score":final_score, "clusters":pred_labels, "pred":pred_ind, 
                    "true":true_ind, "miss_count":miss_count}

    if plot_mode:
        pd_plt.PlotComparisonCombo(true_to_pred, true_series, pred_series)
    return final_score, pred_indices

def AssignmentAccuracy(group_mat, true_mat, pred_mat, true_to_pred, pred_series, pig_count):
    pred_mat = np.zeros((group_mat.shape[0], pig_count), dtype=int)
    for i in range(pig_count):
        pred_mat[:,i] = pred_mat[:,true_to_pred[i]]
    tp = np.sum(true_mat * pred_mat)
    tn = np.sum((1-true_mat) * (1-pred_mat))
    fp = np.sum((1-true_mat) * pred_mat)
    fn = np.sum(true_mat * (1-pred_mat))

    metrics = dict()
    metrics["Accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["Precision"] = (tp / (tp + fp))
    metrics["Recall"] = (tp / (tp + fn))
    metrics["F1 score"] = 2 * (metrics["Precision"] * metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"])
    print("Accuracy:", metrics["Accuracy"], "F1 score:", metrics["F1 score"], "Precision:", metrics["Precision"], "Recall:", metrics["Recall"])
    pd_plt.PlotTrajectoryTruths(true_to_pred, pred_series, true_mat, pred_mat)

def GetSeriesLimits(group_mat):
    max_series = list()
    min_series = list()
    max_value = group_mat[0,0]
    max_series = list([0])
    for i in range(group_mat.shape[0]):
        if group_mat[i,0] > max_value:
            max_value = group_mat[i,0]
            max_series.append(i)

    min_value = group_mat[-1,0]
    min_series = list([group_mat.shape[0]-1])
    for i in reversed(range(group_mat.shape[0])):
        if group_mat[i,0] < min_value:
            min_value = group_mat[i,0]
            min_series.append(i)

    max_series = np.asarray(max_series, dtype=int)
    min_series = np.asarray(min_series, dtype=int)
    max_series = np.take(group_mat, max_series, axis=0)
    min_series = np.flip(np.take(group_mat, min_series, axis=0), axis=0)
    return max_series, min_series

def BenchmarkSimple(estimator, pair_type="full", n_fold=-1, n_threads=10, max_folds=3, pig_count=10):
    with open("data/data_dump.dat", "rb") as fh:
        data_cache = pickle.load(fh)["pig_series"]
    with open("data/data_folds.dat", "rb") as fh:
        fold_sets = pickle.load(fh)[max_folds]
    fold_range = [n_fold] if n_fold != -1 else range(max_folds)
    fold_scores = list()
    for fold_n in fold_range:
        scores = list()
        for series in range(BENCHMARK_SERIES):
            print(series+1, "/", BENCHMARK_SERIES)
            group_mat = SeqGen(group_id=series, pig_count=pig_count, data_cache=data_cache, fold_set=fold_sets[fold_n])
            dist_mat = PrCalc(group_mat, estimator=estimator, pair_type=pair_type, 
                              n_fold=fold_n, n_threads=n_threads, cv_folds=max_folds)
            clustering = HCluster(dist_mat, n_clusters=pig_count, linkage="complete")
            assign_score, _, _ = AssignmentMatch(group_mat, clustering.labels_, pig_count=pig_count, plot_mode=False)
            scores.append(assign_score)
        fold_scores.append([np.mean(scores), np.std(scores)])
    for i, score_set in enumerate(fold_scores):
        print("Model:", estimator, ",Fold:", i+1, ",Mean RMSE:", score_set[0], ",STD RMSE:", score_set[1])

def BenchmarkFull(estimator, pair_type="full", n_fold=-1, n_threads=10, max_folds=3, param_index=-1):
    with open("data/data_dump.dat", "rb") as fh:
        data_cache = pickle.load(fh)["pig_series"]
    with open("data/data_folds.dat", "rb") as fh:
        fold_sets = pickle.load(fh)[max_folds]
    fold_range = [n_fold] if n_fold != -1 else range(max_folds)
    out_dir = "model_benchmarks/" + estimator
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "_indices", exist_ok=True)
    param_combinations = list(itertools.product(*reversed(BENCHMARK_VAR_RANGES)))
    if param_index != -1:
        param_combinations = [param_combinations[param_index]]
    for fold_n in fold_range:
        for series_length, sample_rate, pig_count in param_combinations:
            file_code = str(pig_count) + "-" + str(sample_rate).replace(".",",") + "-" + str(series_length)
            out_name = estimator + "_benchmark_" + file_code + "_" + str(fold_n+1)
            try:
                with open(out_dir + "/" + out_name + ".dat", "rb") as fh:
                    scores = pickle.load(fh)
                with open(out_dir + "_indices/" + out_name + "_indices.dat", "rb") as fh:
                    pred_indices = pickle.load(fh)
                if scores.shape[0] == BENCHMARK_SERIES:
                    continue
                else:
                    scores = list(scores)
                    start_series = len(scores)
            except FileNotFoundError:
                scores = list()
                pred_indices = list()
                start_series = 0
            for series_id in range(start_series, BENCHMARK_SERIES):
                print(fold_n+1, pig_count, sample_rate, series_length, series_id+1, "/", BENCHMARK_SERIES, flush=True)
                group_mat = SeqGen(group_id=series_id, pig_count=pig_count, sample_rate=sample_rate, 
                                data_cache=data_cache, fold_set=fold_sets[fold_n], series_length=series_length)
                dist_mat = PrCalc(group_mat, estimator=estimator, pair_type=pair_type, 
                                  n_fold=fold_n, n_threads=n_threads, cv_folds=max_folds)
                assign_scores_c = list()
                pred_indices_c = list()
                for series_trim in range(BENCHMARK_MIN_TRIM, series_length+1):
                    dist_mat_i = np.copy(dist_mat)
                    group_mat_i = np.copy(group_mat)
                    if series_trim < series_length:
                        group_mat_i = group_mat_i[group_mat_i[:,1] < series_trim * DAY_SECONDS]
                        dist_mat_i = dist_mat_i[:group_mat_i.shape[0],:group_mat_i.shape[0]]
                    if np.unique(group_mat_i[:,2]).shape[0] == pig_count:
                        clustering = HCluster(dist_mat_i, n_clusters=pig_count, linkage="complete")
                        assign_score, pred_indices_c_i = AssignmentMatch(group_mat_i, clustering.labels_, 
                                                                        pig_count=pig_count, plot_mode=False)
                        assign_scores_c.append(assign_score)
                        pred_indices_c.append(pred_indices_c_i)
                    else:
                        assign_scores_c.append(None)
                        pred_indices_c.append(None)
                scores.append(assign_scores_c)
                pred_indices.append(pred_indices_c)
                with open(out_dir + "/" + out_name + ".dat", "wb") as fh:
                    pickle.dump(np.stack(scores), fh)
                with open(out_dir + "_indices/" + out_name + "_indices.dat", "wb") as fh:
                    pickle.dump(pred_indices, fh)

def BenchmarkCompare(n_folds=3):
    mean_results = dict()
    err_results = dict()
    df_dict = defaultdict(list)
    models_full = {"rf-ncv":"Random Forest", "gb-ncv":"Gradient Boosting",
                   "mlp-ncv":"Multilayer Perceptron", "svm_nys-ncv":"SVM Nystroem"}
    for estimator in BENCHMARK_COMPARISON:
        mean_results[estimator] = dict()
        err_results[estimator] = dict()
        for series_length, sample_rate, pig_count in itertools.product(*reversed(BENCHMARK_VAR_RANGES)):
            file_code = str(pig_count) + "-" + str(sample_rate).replace(".",",") + "-" + str(series_length)
            fold_scores = list()
            for fold_n in range(n_folds):
                file_name = estimator + "_benchmark_" + file_code + "_" + str(fold_n+1) + ".dat"
                try:
                    with open("model_benchmarks/" + estimator + "/" + file_name, "rb") as fh:
                        file_in = pickle.load(fh)
                        if len(file_in.shape) > 1:
                            file_in = file_in[:,-1]
                        fold_scores.append(file_in)
                except FileNotFoundError:
                    continue
            if len(fold_scores) == 0:
                continue
            elif len(fold_scores) == 1:
                fold_scores = fold_scores[0]
            else:
                fold_scores = np.concatenate(fold_scores)
            # if np.isinf(fold_scores).any():
            #     fold_scores = fold_scores[np.isinf(fold_scores) == False]
            mean_results[estimator][file_code] = np.mean(fold_scores)
            err_results[estimator][file_code] = np.std(fold_scores) / np.sqrt(fold_scores.shape[0])
            df_dict["Sample Rate"].append(sample_rate)
            df_dict["Pig Count"].append(pig_count)
            df_dict["Model"].append(models_full[estimator])
            df_dict["RMSE (Kg per pig)"].append(mean_results[estimator][file_code])
            df_dict["Standard Error (Kg per pig)"].append(err_results[estimator][file_code])
    var_ranges = BENCHMARK_VAR_RANGES
    var_names = BENCHMARK_VAR_NAMES
    pd_df = pd.DataFrame(data=df_dict)
    pd_df.to_csv("plots/trajectory_comparison.csv")
    pd_plt.PlotBenchmarkComparison(mean_results, err_results, var_names, var_ranges)

def Main():
    parser = argparse.ArgumentParser(description="PIDENT Clustering")
    parser.add_argument("-c", default=10, dest="pig_count", type=int, help="Number of clusters (pigs) (integer)")
    parser.add_argument("-m", default="cluster", dest="mode", choices=["cluster","plot","recalc","benchmark_s","benchmark_f","benchmark_c","plot_raw"], help="Program mode")
    parser.add_argument("-i", default=0, dest="series", type=int, help="Input group series ID")
    parser.add_argument("-n", default="rf-ncv", dest="estimator", help="Distance estimator")
    parser.add_argument("--save", default=False, action="store_true", dest="save_mode", help="Save series (flag only)")
    parser.add_argument("--load", default=False, action="store_true", dest="load_mode", help="Load series (flag only)")
    parser.add_argument("-f", default=-1, dest="n_fold", type=int, help="Benchmark fold (-1 for all)")
    parser.add_argument("-t", default=10, dest="n_threads", type=int, help="Benchmark thread count")
    parser.add_argument("--fold_count", default=3, dest="fold_count", type=int, help="Total outer folds (for -f -1)")
    parser.add_argument("--param_index", default=-1, dest="param_index", type=int, help="Benchmark param index (-1 for all)")
    args = vars(parser.parse_args())
    pig_count = args["pig_count"]

    if args["mode"] == "plot_raw":
        group_mat = SeqGen(group_id=args["series"], pig_count=pig_count)
        pd_plt.PlotRawSeries(group_mat, mode="line")
        sys.exit(0)
    elif args["mode"] == "benchmark_s":
        print("Beginning simple benchmark...")
        BenchmarkSimple(args["estimator"], n_fold=args["n_fold"], n_threads=args["n_threads"], max_folds=args["fold_count"])
        sys.exit(0)
    elif args["mode"] == "benchmark_f":
        print("Beginning full benchmark...")
        BenchmarkFull(args["estimator"], n_fold=args["n_fold"], n_threads=args["n_threads"], 
                      max_folds=args["fold_count"], param_index=args["param_index"])
        sys.exit(0)
    elif args["mode"] == "benchmark_c":
        BenchmarkCompare()
        sys.exit(0)

    if not args["load_mode"]:
        print("Generating sample sequence...")
        group_mat = SeqGen(group_id=args["series"], pig_count=pig_count)
        print("Estimating distances...")
        dist_mat = PrCalc(group_mat, estimator=args["estimator"], pair_type="full")
        print("Estimating connectivity...")
        
        if args["save_mode"]:
            with open("group_series/series_" + str(args["series"]) + "_" + args["estimator"] + ".dat", "wb") as fh:
                pickle.dump((pig_count, group_mat, dist_mat), fh)
    else:
        with open("group_series/series_" + str(args["series"]) + "_" + args["estimator"] + ".dat", "rb") as fh:
            pig_count, group_mat, dist_mat = pickle.load(fh)

    if args["mode"] == "recalc":
        dist_mat = PrCalc(group_mat, estimator=args["estimator"], pair_type="full")
        with open("group_series/series_" + str(args["series"]) + "_" + args["estimator"] + ".dat", "wb") as fh:
            pickle.dump((pig_count, group_mat, dist_mat), fh)
    elif args["mode"] == "plot":
        pd_plt.PlotRawSeries(group_mat)
        pd_plt.PlotClusterMap(dist_mat)
    elif args["mode"] == "dist_plot":
        pd_plt.PlotDistScaling(group_mat, dist_mat)
    elif args["mode"] == "cluster":
        print("Generating clusters...")
        clustering = HCluster(dist_mat, n_clusters=pig_count, linkage="complete")
        # pd_plt.PlotTrajectories(group_mat, clustering.labels_, pig_count, pig_count)
        assign_score, miss_count = AssignmentMatch(group_mat, clustering.labels_, pig_count=pig_count)
        pd_plt.PlotRawSeriesDists(group_mat, dist_mat, clustering.labels_)
        print("RMSE score:", assign_score, "Miss count:", miss_count)
    else:
        raise RuntimeError("Unknown mode: " + args["mode"])

if __name__ == "__main__":
    Main()