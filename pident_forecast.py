import sys
import os
import pickle
import argparse
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GroupKFold
from scipy.optimize import linear_sum_assignment
from util.forecast_util import GenerateDataset

def TrainCV(dataset, model_name, std_pred=False):
    if not std_pred:
        data_x = np.copy(dataset["x"])
    else:
        data_x = np.copy(dataset["x"][:,:-1])
    x_scaler = StandardScaler()
    data_x = x_scaler.fit_transform(data_x)
    data_y = np.copy(dataset["y"]) / 1000
    cv_folds = PredefinedSplit(dataset["folds"])

    if model_name == "rf":
        rgs = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=555)
    elif model_name == "gb":
        rgs = GradientBoostingRegressor(random_state=555)
        if std_pred:
            rgs_std = GradientBoostingRegressor(random_state=555)
            data_y_mean = data_y[:,0]
            data_y_std = data_y[:,1]
    elif model_name == "lasso":
        if not std_pred:
            rgs = Lasso(max_iter=10000, random_state=555)
        else:
            rgs = MultiTaskLasso(random_state=555)
    elif model_name == "dummy":
        rgs = DummyRegressor()
    
    cv_out = defaultdict(list)
    if std_pred and model_name == "gb":
        cv_results_mean = cross_validate(rgs, data_x, data_y_mean, scoring="neg_root_mean_squared_error",
                                    n_jobs=3, return_estimator=True, cv=cv_folds)
        cv_results_std = cross_validate(rgs_std, data_x, data_y_std, scoring="neg_root_mean_squared_error",
                                    n_jobs=3, return_estimator=True, cv=cv_folds)
        cv_out["estimators"] = list()
        cv_out["validation_fold_rmse"] = list()
        for i in range(len(cv_results_mean["estimator"])):
            cv_out["estimators"].append([cv_results_mean["estimator"][i], cv_results_std["estimator"][i]])
            cv_out["validation_fold_rmse"].append((cv_results_mean["test_score"][0] + cv_results_std["test_score"][0]) * -0.5)
        cv_out["validation_mean_rmse"] = np.mean(cv_out["validation_fold_rmse"])
    else:
        cv_results = cross_validate(rgs, data_x, data_y, scoring="neg_root_mean_squared_error",
                            n_jobs=3, return_estimator=True, cv=cv_folds)
        cv_out["estimators"] = cv_results["estimator"]
        cv_out["validation_mean_rmse"] = np.mean(cv_results["test_score"])*-1
        cv_out["validation_fold_rmse"] = [x*-1 for x in cv_results["test_score"]]

    cv_out["model_name"] = model_name
    cv_out["scaler"] = x_scaler
    fold_n = 0
    for train_ix, test_ix in cv_folds.split():
        x_test, y_test = data_x[test_ix], data_y[test_ix]
        if std_pred and model_name == "gb":
            y_pred_mean = cv_out["estimators"][fold_n][0].predict(x_test)
            y_pred_std = cv_out["estimators"][fold_n][1].predict(x_test)
            y_pred = np.stack((y_pred_mean, y_pred_std), axis=1)
        else:
            y_pred = cv_out["estimators"][fold_n].predict(x_test)
        cv_out["pred"].append(y_pred)
        cv_out["true"].append(y_test)
        cv_out["groups"].append(dataset["groups"][test_ix])
        cv_out["steps"].append(dataset["steps"][test_ix])
        if std_pred:
            cv_out["counts"].append(dataset["x"][test_ix,-1])
        fold_n += 1
    return cv_out

def TrainNCV(dataset, model_name, std_pred=False, n_threads=5):
    if not std_pred:
        data_x = np.copy(dataset["x"])
    else:
        data_x = np.copy(dataset["x"][:,:-1])
    x_scaler = StandardScaler()
    data_x = x_scaler.fit_transform(data_x)
    data_y = np.copy(dataset["y"]) / 1000

    nested_out = defaultdict(list)
    nested_out["model_name"] = model_name
    nested_out["scaler"] = x_scaler

    cv_outer = PredefinedSplit(dataset["folds"])
    current_fold = 1
    fold_count = np.amax(dataset["folds"]) + 1
    for train_ix, test_ix in cv_outer.split():
        print("Starting fold", current_fold, "/", fold_count)
        X_train, X_test = data_x[train_ix, :], data_x[test_ix, :]
        y_train, y_test = data_y[train_ix], data_y[test_ix]
        
        if model_name == "rf":
            rgs = RandomForestRegressor(random_state=555)
            param_grid = {"n_estimators":[50,75,100,125,150,175,200],
                          "max_depth":[5,10,15,20,25,30,35]}
        elif model_name == "gb":
            rgs = GradientBoostingRegressor(random_state=555)
            if std_pred:
                rgs_std = GradientBoostingRegressor(random_state=555)
                y_train_mean = y_train[:,0]
                y_train_std = y_train[:,1]
            param_grid = {"n_estimators":[50,75,100,125,150,175,200],
                          "max_depth":[2,4,6,8,10,12]}
        elif model_name == "dummy":
            rgs = DummyRegressor()

        if std_pred and model_name == "gb":
            cv_inner_mean = GroupKFold(n_splits=5).split(X_train, y_train_mean, dataset["groups"][train_ix])
            cv_inner_std = GroupKFold(n_splits=5).split(X_train, y_train_std, dataset["groups"][train_ix])
        else:
            cv_inner = GroupKFold(n_splits=5).split(X_train, y_train, dataset["groups"][train_ix])

        if model_name != "lasso":
            if std_pred and model_name == "gb":
                gridsearch_mean = GridSearchCV(rgs, param_grid, scoring="neg_root_mean_squared_error", 
                                        cv=cv_inner_mean, n_jobs=n_threads, refit=True)
                gridsearch_std = GridSearchCV(rgs_std, param_grid, scoring="neg_root_mean_squared_error", 
                                        cv=cv_inner_std, n_jobs=n_threads, refit=True)
                gs_result_mean = gridsearch_mean.fit(X_train, y_train_mean)
                gs_result_std = gridsearch_std.fit(X_train, y_train_std)
                best_model = [gs_result_mean.best_estimator_, gs_result_std.best_estimator_]
                nested_out["best_params"].append([gs_result_mean.best_params_, gs_result_std.best_params_])
                nested_out["gs_results"].append([gs_result_mean.cv_results_, gs_result_std.cv_results_])
            else:
                gridsearch = GridSearchCV(rgs, param_grid, scoring="neg_root_mean_squared_error", 
                                        cv=cv_inner, n_jobs=n_threads, refit=True)
                gs_result = gridsearch.fit(X_train, y_train)
                best_model = gs_result.best_estimator_
                nested_out["best_params"].append(gs_result.best_params_)
                nested_out["gs_results"].append(gs_result.cv_results_)
        else:
            if std_pred:
                gridsearch = MultiTaskLassoCV(max_iter=10000, random_state=555)
                gs_result = gridsearch.fit(X_train, y_train)
                best_model = MultiTaskLasso(alpha=gs_result.alpha_, max_iter=10000, random_state=555)
            else:
                gridsearch = LassoCV(max_iter=10000, random_state=555)
                gs_result = gridsearch.fit(X_train, y_train)
                best_model = Lasso(alpha=gs_result.alpha_, max_iter=10000, random_state=555)
            best_model.fit(X_train, y_train)
            nested_out["best_params"].append({"alpha":gs_result.alpha_})

        if std_pred and model_name == "gb":
            y_pred_mean = best_model[0].predict(X_test)
            y_pred_std = best_model[1].predict(X_test)
            y_pred = np.stack((y_pred_mean, y_pred_std), axis=1)
        else:
            y_pred = best_model.predict(X_test)
        model_score = np.sqrt(np.mean((y_pred - y_test) ** 2))

        nested_out["pred"].append(y_pred)
        nested_out["true"].append(y_test)
        nested_out["groups"].append(dataset["groups"][test_ix])
        nested_out["steps"].append(dataset["steps"][test_ix])
        nested_out["validation_fold_rmse"].append(model_score)
        nested_out["estimators"].append(best_model)
        if std_pred:
            nested_out["counts"].append(dataset["x"][test_ix,-1])
        current_fold += 1
    
    print("Optimal parameters:", nested_out["best_params"])
    nested_out["validation_mean_rmse"] = np.mean(nested_out["validation_fold_rmse"])
    return nested_out

def TestCV(dataset, model, std_pred=False):
    if not std_pred:
        data_x = np.copy(dataset["x"])
    else:
        data_x = np.copy(dataset["x"][:,:-1])
    data_x = model["scaler"].transform(data_x)
    data_y = np.copy(dataset["y"]) / 1000
    cv_folds = PredefinedSplit(dataset["folds"])

    test_out = defaultdict(list)
    fold_n = 0
    for train_ix, test_ix in cv_folds.split():
        x_test, y_test = data_x[test_ix], data_y[test_ix]
        if std_pred and model["model_name"] == "gb":
            y_pred_mean = model["estimators"][fold_n][0].predict(x_test)
            y_pred_std = model["estimators"][fold_n][1].predict(x_test)
            y_pred = np.stack((y_pred_mean, y_pred_std), axis=1)
        else:
            y_pred = model["estimators"][fold_n].predict(x_test)
        model_score = np.sqrt(np.mean((y_pred - y_test) ** 2))
        test_out["test_fold_rmse"].append(model_score)
        if std_pred:
            model_score_meanpred = np.sqrt(np.mean((y_pred[:,0] - y_test[:,0]) ** 2))
            model_score_stdpred = np.sqrt(np.mean((y_pred[:,1] - y_test[:,1]) ** 2))
            test_out["test_fold_rmse_meanpred"].append(model_score_meanpred)
            test_out["test_fold_rmse_stdpred"].append(model_score_stdpred)
        fold_n += 1
    test_out["test_mean_rmse"] = np.mean(test_out["test_fold_rmse"])
    if std_pred:
        test_out["test_mean_rmse_meanpred"] = np.mean(test_out["test_fold_rmse_meanpred"])
        test_out["test_mean_rmse_stdpred"] = np.mean(test_out["test_fold_rmse_stdpred"])
    return test_out

def MatchIndividualWeights(model):
    fold_scores = list()
    all_scores = list()
    for true, pred, groups, steps in zip(model["true"], model["pred"], 
                                         model["groups"], model["steps"]):
        group_scores = list()
        for group_i in np.unique(groups):
            group_indices = groups == group_i
            for step_i in np.unique(steps[group_indices]):
                step_indices = np.logical_and(group_indices, steps == step_i)
                true_step = true[step_indices]
                pred_step = pred[step_indices]
                dist_mat = np.zeros((true_step.shape[0], pred_step.shape[0]), dtype=float)
                for i in range(true_step.shape[0]):
                    for j in range(pred_step.shape[0]):
                        dist_mat[i,j] = (true_step[i] - pred_step[j]) ** 2
                true_indices, pred_indices = linear_sum_assignment(dist_mat)
                group_scores.append(dist_mat[true_indices,pred_indices].flatten())
        all_scores += group_scores
        group_scores = np.concatenate(group_scores)
        fold_scores.append(np.sqrt(np.mean(group_scores)))
    print("Matched individual RMSE fold scores:", fold_scores)
    print("Matched individual RMSE average score:", np.mean(fold_scores))
    all_scores = np.concatenate(all_scores)
    all_scores = np.sort(all_scores)
    print(all_scores[-10:])
    return fold_scores

def EstimateIndividualWeights(model, indv_dataset):
    rng = np.random.Generator(np.random.PCG64(555))

    cv_outer = PredefinedSplit(indv_dataset["folds"])
    indv_true_y = list()
    indv_true_groups = list()
    indv_true_steps = list()
    for train_ix, test_ix in cv_outer.split():
        indv_true_y.append(indv_dataset["y"][test_ix])
        indv_true_groups.append(indv_dataset["groups"][test_ix])
        indv_true_steps.append(indv_dataset["steps"][test_ix])

    fold_scores = list()
    fold_sizes = list()
    model_iter = zip(indv_true_y, indv_true_groups, indv_true_steps, 
                     model["pred"], model["groups"], model["steps"], model["counts"])
    for true, true_groups, true_steps, pred, pred_groups, pred_steps, pred_counts in model_iter:
        group_scores = list()
        for group_i in np.unique(pred_groups):
            group_indices_true = true_groups == group_i
            group_indices_pred = pred_groups == group_i
            for step_i in np.unique(pred_steps[group_indices_pred]):
                step_indices_true = np.logical_and(group_indices_true, true_steps == step_i)
                step_indices_pred = np.logical_and(group_indices_pred, pred_steps == step_i)
                true_step = true[step_indices_true] / 1000
                pred_step_mean = pred[step_indices_pred,0]
                pred_step_std = pred[step_indices_pred,1]
                pred_count = round(pred_counts[step_indices_pred][0])
                pred_step = rng.normal(loc=pred_step_mean, scale=pred_step_std, size=pred_count)
                dist_mat = np.zeros((true_step.shape[0], pred_step.shape[0]), dtype=float)
                assert true_step.shape[0] == pred_step.shape[0]

                for i in range(true_step.shape[0]):
                    for j in range(pred_step.shape[0]):
                        dist_mat[i,j] = (true_step[i] - pred_step[j]) ** 2
                true_indices, pred_indices = linear_sum_assignment(dist_mat)
                group_scores.append(dist_mat[true_indices,pred_indices])
        group_scores = np.concatenate(group_scores)
        fold_scores.append(np.sqrt(np.mean(group_scores)))
        fold_sizes.append(group_scores.shape[0])
    print("Estimated individual RMSE fold scores:", fold_scores)
    print("Estimated individual RMSE average score:", np.mean(fold_scores))
    return fold_scores

def ModelComparison(nested_cv=False):
    figure_colours = ["red", "blue", "limegreen", "purple", "orange", 
                      "brown", "darkred", "darkblue", "green"]
    format_modes = ["true", "pred", "group"]
    pig_counts = [10, 20, 30, 40, 50]
    sample_rates = [1.0, 2.0, 3.0, 4.0]
    models = ["lasso", "rf", "gb"]
    models_full = {"lasso":"LASSO", "rf":"Random Forest", "gb":"Gradient Boosting"}
    validation_type = "ncv" if nested_cv else "cv"
    fig, ax = plt.subplots(figsize=(12,8), nrows=2, ncols=2)
    ax_l = ax.reshape(-1)
    all_scores = list()
    df_dict = defaultdict(list)
    type_scores = defaultdict(list)

    for plot_i, sample_rate in enumerate(sample_rates):
        series_i = 0
        for format_mode, model in itertools.product(*(format_modes, models)):
            score_series = list()
            pig_counts_series = list()
            df_dict["Sample Rate"].append(sample_rate)
            # df_dict["Pig count"].append(pig_count)
            df_dict["Input Type"].append(format_mode.capitalize())
            df_dict["Model"].append(models_full[model])
            for pig_count in pig_counts:
                file_code = "_" + str(pig_count) + "-" + str(sample_rate).replace(".",",")
                model_name = validation_type + "_" + model + "_" + format_mode
                try:
                    with open("model_states_forecast/" + model_name + file_code + ".dat", "rb") as fh:
                        model_data = pickle.load(fh)
                except FileNotFoundError:
                    print("Missing model:", model_name + file_code)
                    continue
                # print(model, model_data["best_params"])
                score_series.append(model_data["eval_mean_rmse"])
                all_scores.append(model_data["eval_mean_rmse"])
                pig_counts_series.append(pig_count)
                df_dict[str(pig_count) + "-pig RMSE (Kg)"].append(model_data["eval_mean_rmse"])
                type_scores[format_mode].append(model_data["eval_mean_rmse"])
            plot_label = format_mode + "_" + model
            ax_l[plot_i].plot(pig_counts_series, score_series, c=figure_colours[series_i], label=plot_label, marker="x")
            ax_l[plot_i].title.set_text("Sample rate: " + str(sample_rate))
            ax_l[plot_i].set_xlim((10-2, 50+2))
            ax_l[plot_i].yaxis.grid(True)
            series_i += 1
    
    min_score = min(all_scores)
    max_score = max(all_scores)
    for plot_i in range(ax_l.shape[0]):
        ax_l[plot_i].set_ylim(min_score - 0.5, max_score + 0.5)

    plt.setp(ax, xticks=[10,20,30,40,50])
    fig.text(0.46, 0.02, "Pig count", ha="center", va="center", fontsize="medium")
    fig.text(0.02, 0.5, "RMSE (Kg)", ha="center", va="center", rotation="vertical", fontsize="medium")
    handles, labels = ax_l[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=7, title="Model", fontsize="medium", frameon=False)
    plt.subplots_adjust(left=0.05, right=0.87, top=0.95, bottom=0.06)
    plt.show()

    pd_df = pd.DataFrame(data=df_dict)
    # true_avg = pd_df[pd_df["Input type"] == "True"]["RMSE (Kg)"].mean()
    true_avg = np.mean(type_scores["true"])
    pred_avg = np.mean(type_scores["pred"])
    group_avg = np.mean(type_scores["group"])
    print("True average score:", true_avg, "Pred average score:", pred_avg, "Group average score:", group_avg)
    print("Group - pred:", group_avg - pred_avg, "Pred - true:", pred_avg - true_avg)
    pd_df.to_csv("plots/forecast_comparison.csv")
    sys.exit(0)

def Main():
    parser = argparse.ArgumentParser(description="PIDENT Forecasting")
    parser.add_argument("-f", default="true", dest="format_mode", choices=["true","pred","group"], help="Trajectory format mode")
    parser.add_argument("-c", default=10, dest="pig_count", type=int, help="Number of pigs")
    parser.add_argument("-r", default=1.0, dest="sample_rate", choices=[1.0,2.0,3.0,4.0], type=float, help="Sample rate")
    parser.add_argument("-s", default=False, action="store_true", dest="save_out", help="Save model performance (flag only)")
    parser.add_argument("--compare", default=False, action="store_true", dest="compare", help="Model comparison mode (flag only)")

    parser.add_argument("-m", default="rf", dest="model", choices=["rf","gb","lasso","dummy"], help="Prediction model")
    parser.add_argument("-n", default=False, action="store_true", dest="nested_cv", help="Nested CV mode (flag only)")
    parser.add_argument("-t", default=3, dest="n_threads", type=int, help="Number of CPU threads")
    parser.add_argument("--outer_folds", default=3, dest="outer_folds", type=int, help="Outer CV folds")

    parser.add_argument("--tj", default="rf-ncv", dest="tj_model", help="Trajectory model (pred mode)")
    parser.add_argument("--series_count", default=100, dest="series_count", type=int, help="Number of series")
    parser.add_argument("--series_length", default=35, dest="series_length", type=int, help="Series length")

    parser.add_argument("--step_size", default=1.0, dest="step_size", type=float, help="Step size (days)")
    parser.add_argument("--horizon", default=14, dest="horizon", type=int, help="Forecast horizon (steps)")
    parser.add_argument("--frag_size", default=7, dest="frag_size", type=int, help="Fragment size (steps)")
    parser.add_argument("--min_pigs", default=1.0, dest="min_pigs", type=float, help="Minimum pigs ratio (float)")
    args = vars(parser.parse_args())

    if args["compare"]:
        ModelComparison(nested_cv=args["nested_cv"])

    print("Generating dataset...")
    dataset = GenerateDataset(args["format_mode"], args["tj_model"],
                              series_count=args["series_count"],
                              pig_count=args["pig_count"],
                              sample_rate=args["sample_rate"],
                              series_length=args["series_length"],
                              fold_count=args["outer_folds"],
                              min_pigs=args["min_pigs"],
                              ff_step_size=args["step_size"],
                              ff_frag_size=args["frag_size"],
                              ff_horizon=args["horizon"])

    std_pred = args["format_mode"] == "group"
    if args["nested_cv"]:
        print("\nNested cross-validation started...")
        cv_train = TrainNCV(dataset, args["model"], n_threads=args["n_threads"], std_pred=std_pred)
    else:
        print("\nCross-validation started...")
        cv_train = TrainCV(dataset, args["model"], std_pred=std_pred)

    cv_test = TestCV(dataset, cv_train, std_pred=std_pred)
    cv_test["validation_fold_rmse"] = cv_train["validation_fold_rmse"]
    cv_test["validation_mean_rmse"] = cv_train["validation_mean_rmse"]

    if args["nested_cv"]:
        cv_test["best_params"] = cv_train["best_params"]
        if args["model"] != "lasso":
            cv_test["gs_results"] = cv_train["gs_results"]

    if args["format_mode"] == "group":
        print("RMSE fold scores:", cv_test["test_fold_rmse"])
        print("RMSE average score:", cv_test["test_mean_rmse"], "\n")
        print("Mean prediction RMSE fold scores:", cv_test["test_fold_rmse_meanpred"])
        print("Mean prediction RMSE average score:", cv_test["test_mean_rmse_meanpred"], "\n")
        print("STD prediction RMSE fold scores:", cv_test["test_fold_rmse_stdpred"])
        print("STD prediction RMSE average score:", cv_test["test_mean_rmse_stdpred"], "\n")
        print("Evaluation started - loading individual-level dataset...")
        dataset_indv = GenerateDataset("true", args["tj_model"],
                                       series_count=args["series_count"],
                                       pig_count=args["pig_count"],
                                       sample_rate=args["sample_rate"],
                                       series_length=args["series_length"],
                                       fold_count=args["outer_folds"],
                                       min_pigs=args["min_pigs"],
                                       ff_step_size=args["step_size"],
                                       ff_frag_size=args["frag_size"],
                                       ff_horizon=args["horizon"])
        eval_scores = EstimateIndividualWeights(cv_train, dataset_indv)
    else:
        eval_scores = MatchIndividualWeights(cv_train)
    cv_test["eval_fold_rmse"] = eval_scores
    cv_test["eval_mean_rmse"] = np.mean(eval_scores)

    if args["save_out"]:
        cv_test["args"] = args
        file_code = "_" + str(args["pig_count"]) + "-" + str(args["sample_rate"]).replace(".",",")
        validation_type = "ncv" if args["nested_cv"] else "cv"
        model_name = validation_type + "_" + args["model"] + "_" + args["format_mode"]
        os.makedirs("model_states_forecast", exist_ok=True)
        with open("model_states_forecast/" + model_name + file_code + ".dat", "wb") as fh:
            pickle.dump(cv_test, fh)

if __name__ == "__main__":
    Main()
