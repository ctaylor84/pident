import pickle
import argparse
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA

def GetModel(model_name, svm_kernel="rbf", svm_c=1.0):
    name_split = model_name.split("_")
    model_type = name_split[0]
    try:
        pred_type = name_split[1]
    except IndexError:
        pred_type = "rgs"
    
    if pred_type == "pr":
        pr_type = True
        if model_type == "svm":
            if svm_kernel == "linear":
                clf = LinearSVC(C=svm_c, random_state=555, max_iter=10000)
            elif svm_kernel == "nystroem":
                clf = Pipeline([("nystroem", Nystroem(random_state=555)), 
                                ("svc", LinearSVC(C=svm_c, random_state=555))])
            else:
                clf = SVC(C=svm_c, kernel=svm_kernel, cache_size=1500, random_state=555)
        elif model_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(64,64,64), batch_size=512, random_state=555)
        elif model_type == "rf":
            clf = RandomForestClassifier(max_depth=20, n_jobs=1, random_state=555)
        elif model_type == "gb":
            clf = GradientBoostingClassifier(random_state=555)
        else:
            raise RuntimeError("Unknown model type")
    else:
        pr_type = False
        if model_type == "svm":
            if svm_kernel == "linear":
                clf = LinearSVC(C=svm_c, random_state=555, max_iter=10000)
            elif svm_kernel == "nystroem":
                clf = Pipeline([("nystroem", Nystroem(random_state=555)), 
                                ("svc", LinearSVC(C=svm_c, random_state=555))])
            else:
                clf = SVC(C=svm_c, kernel=svm_kernel, cache_size=1500, random_state=555)
            pr_type = True
        elif model_type == "mlp":
            clf = MLPRegressor(hidden_layer_sizes=(64,64,64), batch_size=512, random_state=555)
        elif model_type == "rf":
            clf = RandomForestRegressor(max_depth=20, n_jobs=1, random_state=555)
        elif model_type == "gb":
            clf = GradientBoostingRegressor(random_state=555)
        else:
            raise RuntimeError("Unknown model type")
    return clf, pr_type

def PrTrain(model_name, pair_type="full", svm_kernel="rbf"):
    pairs = np.load("data/data_pairs_" + pair_type + ".npy")
    x_all = np.stack((pairs[:,0], pairs[:,2], 
                      pairs[:,1] - pairs[:,0]), axis=1)
    x_all = StandardScaler().fit_transform(x_all)
    y_all = pairs[:,3]

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.95, random_state=555)
    print("Train size:", x_train.shape[0])
    print("Test size:", x_test.shape[0])

    clf, pr_mode = GetModel(model_name, svm_kernel=svm_kernel)
    clf.fit(x_train, y_train)
    with open("model_states/" + model_name + ".dat", "wb") as fh:
        pickle.dump(clf, fh)

def LoadFolds(n_folds, pair_type="full", shuffle=True, reduction=None):
    with open("data/data_pairs_" + str(n_folds) + "fold_" + pair_type + ".dat", "rb") as fh:
        fold_data = pickle.load(fh)

    split_indices = list()
    for n_fold in range(n_folds):
        split_indices.append(np.repeat(n_fold, len(fold_data[n_fold])))
    full_data = np.concatenate(fold_data)
    split_indices = np.concatenate(split_indices)

    if reduction != None:
        rng = np.random.Generator(np.random.PCG64(555))
        pos_indices = np.squeeze(np.argwhere(np.around(full_data[:,3]) == 1))
        neg_indices = np.squeeze(np.argwhere(np.around(full_data[:,3]) == 0))
        pos_indices = rng.choice(pos_indices, size=int(pos_indices.shape[0] * reduction), replace=False)
        neg_indices = rng.choice(neg_indices, size=int(neg_indices.shape[0] * reduction), replace=False)
        reduct_indices = np.concatenate((pos_indices, neg_indices))
        full_data = full_data[reduct_indices]
        split_indices = split_indices[reduct_indices]

    if shuffle:
        rng = np.random.Generator(np.random.PCG64(555))
        new_indices = np.arange(full_data.shape[0])
        rng.shuffle(new_indices)
        full_data = full_data[new_indices]
        split_indices[new_indices]

    cv_folds = PredefinedSplit(split_indices)
    return full_data, cv_folds

def PrCrossVal(model_name, pair_type="full", n_folds=3, svm_kernel="rbf", svm_c=1.0):
    if model_name[:3] != "svm":
        pairs, cv_folds = LoadFolds(n_folds, pair_type=pair_type)
    else:
        pairs, cv_folds = LoadFolds(n_folds, pair_type=pair_type, reduction=0.25)
    
    x_all = np.stack((pairs[:,0], pairs[:,2], 
                      pairs[:,1] - pairs[:,0]), axis=1)

    scaler = StandardScaler().fit(x_all)
    with open("model_states/scaler_" + model_name + "_" + pair_type + ".dat", "wb") as fh:
        pickle.dump(scaler, fh)

    x_all = scaler.transform(x_all)
    y_all = pairs[:,3]

    clf, pr_mode = GetModel(model_name, svm_kernel=svm_kernel, svm_c=svm_c)
    if pr_mode:
        cv_results = cross_validate(clf, x_all, y_all, scoring=["average_precision","precision","recall"], 
                                    n_jobs=5, return_estimator=True, cv=cv_folds)
        print(cv_results)
    else:
        cv_results = cross_validate(clf, x_all, y_all, scoring=["neg_root_mean_squared_error"], 
                                    n_jobs=5, return_estimator=True, cv=cv_folds)
        cv_scores = cv_results["test_neg_root_mean_squared_error"] * -1.0
        print("RMSE Scores:", cv_scores)
        print("RMSE Mean:", np.mean(cv_scores))
    with open("model_states/cv_" + model_name + "_" + pair_type + ".dat", "wb") as fh:
        pickle.dump(cv_results, fh)

def GetParamGrid(model_name, svm_kernel=None):
    model_type = model_name.split("_")[0]
    if model_type == "svm":
        if svm_kernel == "nystroem":
            param_grid = {"svc__C":[0.1,0.5,1.0,5.0,10.0],
                          "nystroem__n_components":[25,50,75,100]}
        else:
            param_grid = {"C":[0.1,1.0,10.0]}
    elif model_type == "mlp":
        param_grid = {"hidden_layer_sizes":[(32,32),(64,64),(128,128),(256,256), #31 min per core
                      (32,32,32),(64,64,64),(128,128,128),(256,256,256),
                      (32,32,32,32),(64,64,64,64),(128,128,128,128),(256,256,256,256),
                      (32,32,32,32,32),(64,64,64,64,64),(128,128,128,128,128),(256,256,256,256,256)]}
    elif model_type == "rf":
        param_grid = {"n_estimators":[100,125,150,175,200],
                      "max_depth":[10,15,20,25,30]} #10 min per core
    elif model_type == "gb":
        param_grid = {"n_estimators":[100,150,200,250,300],
                      "max_depth":[12,15,18,21]} #5 min per core
    return param_grid

def NestedCV(model_name, pair_type="full", num_threads=5, outer_folds=3, 
             split_id=0, svm_kernel="rbf"):
    pairs, cv_outer = LoadFolds(outer_folds, pair_type=pair_type)
    x_all = np.stack((pairs[:,0], pairs[:,2], 
                      pairs[:,1] - pairs[:,0]), axis=1)
    scaler = StandardScaler().fit(x_all)
    x_all = scaler.transform(x_all)
    y_all = pairs[:,3]

    param_grid = GetParamGrid(model_name, svm_kernel=svm_kernel)

    if split_id == 0:
        outer_splits = cv_outer.split(x_all, y_all)
        outer_scores = list()
        best_estimators = list()
        grid_search_results = list()
        current_fold = 1
    else:
        outer_splits = [list(cv_outer.split(x_all, y_all))[split_id-1]]
    
    for train_ix, test_ix in outer_splits:
        if split_id == 0:
            print("Starting fold", current_fold, "/", outer_folds)
        X_train, X_test = x_all[train_ix, :], x_all[test_ix, :]
        y_train, y_test = y_all[train_ix], y_all[test_ix]

        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=555)
        model, pr_type = GetModel(model_name, svm_kernel=svm_kernel)

        if pr_type:
            metric = "average_precision"
        else:
            metric = "neg_root_mean_squared_error"

        gridsearch = GridSearchCV(model, param_grid, scoring=metric, cv=cv_inner, n_jobs=num_threads, refit=True)
        gs_result = gridsearch.fit(X_train, y_train)
        best_model = gs_result.best_estimator_
        y_pred = best_model.predict(X_test)
        if pr_type:
            model_score = average_precision_score(y_test, y_pred)
        else:
            model_score = np.sqrt(np.mean((y_pred - y_test) ** 2))

        if split_id == 0:
            outer_scores.append(model_score)
            best_estimators.append(best_model)
            grid_search_results.append(gs_result.cv_results_)
            current_fold += 1
        else:
            outer_score = model_score
            best_estimator = best_model
            grid_search_results = gs_result.cv_results_
    nested_out = dict()
    nested_out["model_name"] = model_name
    nested_out["scaler"] = scaler
    nested_out["pr_type"] = pr_type
    nested_out["gs_results"] = grid_search_results

    if split_id == 0:
        if pr_type:
            print("Average precision fold scores:", outer_scores)
            print("Average precision average score:", np.mean(outer_scores))
        else:
            print("RMSE fold scores:", outer_scores)
            print("RMSE average score:", np.mean(outer_scores))
        nested_out["scores"] = outer_scores
        nested_out["estimators"] = best_estimators
        with open("model_states_ncv/ncv_" + model_name + "_" + pair_type + ".dat", "wb") as fh:
            pickle.dump(nested_out, fh)
    else:
        nested_out["fold_id"] = split_id
        nested_out["score"] = outer_score
        nested_out["estimator"] = best_estimator
        with open("model_states_ncv/ncv_" + model_name + "_" + pair_type + "_" + str(split_id) + ".dat", "wb") as fh:
            pickle.dump(nested_out, fh)

def PrCrossValResults(model_name, pair_type="full"):
    with open("model_states/cv_" + model_name + "_" + pair_type + ".dat", "rb") as fh:
        cv_results = pickle.load(fh)
    print(cv_results)

def PrEstimateHelper(id, x, estimator, out_queue, dist_func):
    if not dist_func:
        # scores = estimator.predict_proba(x)
        scores = estimator.predict(x)
    else:
        scores = estimator.decision_function(x)
    out_queue.put([id,scores])

def LoadModel(model_name, pair_type="full", cv_folds=3):
    if model_name[-4:] != "-ncv":
        with open("model_states/cv_" + model_name + "_" + pair_type + ".dat", "rb") as fh:
            estimators = pickle.load(fh)["estimator"]
        with open("model_states/scaler_" + model_name + "_" + pair_type + ".dat", "rb") as fh:
            scaler = pickle.load(fh)
    else:
        model_name = model_name[:-4]
        try:
            with open("model_states_ncv/ncv_" + model_name + "_" + pair_type + ".dat", "rb") as fh:
                ncv_data = pickle.load(fh)
            estimators = ncv_data["estimators"]
            scaler = ncv_data["scaler"]
        except FileNotFoundError:
            estimators = list()
            for n_fold in range(1, cv_folds+1):
                with open("model_states_ncv/ncv_" + model_name + "_" + pair_type + "_" + str(n_fold) + ".dat", "rb") as fh:
                    ncv_data = pickle.load(fh)
                estimators.append(ncv_data["estimator"])
                scaler = ncv_data["scaler"]
    return estimators, scaler

def PrEstimate(pairs, model_name="svm_rbf", pair_type="full", cv_folds=3, 
               n_fold=-1, n_threads=10):
    estimators, scaler = LoadModel(model_name, pair_type=pair_type, cv_folds=cv_folds)

    x_all = np.stack((pairs[:,0], pairs[:,2], 
                      pairs[:,1] - pairs[:,0]), axis=1)
    # x_all = np.stack((pairs[:,0], pairs[:,2],
    #                   pairs[:,1] - pairs[:,0],
    #                  (pairs[:,1] - pairs[:,0]) / pairs[:,2]), axis=1)
    x_all = scaler.transform(x_all)

    if model_name.split("_")[0] == "svm":
        dist_func = True
    else:
        dist_func = False

    processes = list()
    out_queue = mp.Queue()
    if n_fold == -1:
        for clf in estimators:
            p = mp.Process(target=PrEstimateHelper, args=(0, x_all, clf, out_queue, dist_func))
            p.start()
            processes.append(p)
        scores = list()
        for p in processes:
            scores.append(out_queue.get(True)[1])
        for p in processes:
            p.join()
        scores = np.stack(scores)
        scores = np.mean(scores, axis=0)
    else:
        clf = estimators[n_fold]
        process_id = 0
        chunk_range = list(np.around(np.linspace(0, x_all.shape[0], n_threads+1)).astype(int))[:-1]
        for i in range(len(chunk_range)-1):
            p = mp.Process(target=PrEstimateHelper, args=(process_id, x_all[chunk_range[i]:chunk_range[i+1]], clf, out_queue, dist_func))
            p.start()
            processes.append(p)
            process_id += 1
        p = mp.Process(target=PrEstimateHelper, args=(process_id, x_all[chunk_range[-1]:], clf, out_queue, dist_func))
        p.start()
        processes.append(p)
        scores = list([0]*n_threads)
        for p in processes:
            queue_out = out_queue.get(True)
            scores[queue_out[0]] = queue_out[1]
        for p in processes:
            p.join()
        scores = np.concatenate(scores)
    assert scores.shape[0] == x_all.shape[0]
    return scores

def PrMapHelper(x, estimator, dist_func):
    if not dist_func:
        # scores = estimator.predict_proba(x)
        scores = estimator.predict(x)
    else:
        scores = estimator.decision_function(x)
    return scores

def PlotMap(single_model=None, pair_type="full", cv_folds=3, n_threads=10):
    pairs, cv_outer = LoadFolds(cv_folds, pair_type=pair_type)
    x_all = np.stack((pairs[:,0], pairs[:,2], 
                      pairs[:,1] - pairs[:,0]), axis=1)

    pca_scaler = StandardScaler()
    pca = PCA(n_components=2)
    pca.fit(pca_scaler.fit_transform(x_all))
    x_all_pca = pca.transform(pca_scaler.transform(x_all))
    print("PCA variance:", pca.explained_variance_ratio_)
    var_max = np.amax(x_all_pca, axis=1)
    var_min = np.amin(x_all_pca, axis=1)

    graph_res = 100
    xg = np.meshgrid(np.linspace(var_min[0], var_max[0], graph_res), 
                     np.linspace(var_min[1], var_max[1], graph_res))
    x_all_pca = np.transpose(xg).reshape(-1,2)
    x_all_pcainv = pca_scaler.inverse_transform(pca.inverse_transform(x_all_pca))
    chunk_range = list(np.around(np.linspace(0, x_all_pcainv.shape[0], n_threads+1)).astype(int))

    scores_all = list()
    if single_model == None:
        models = ["svm_rbf", "svm_nys-ncv", "gb-ncv", "gb_pr-ncv", "mlp-ncv", 
                  "mlp_pr-ncv", "rf-ncv", "rf_pr-ncv"]
    else:
        models = [single_model]
    
    for model_name in models:
        estimators, scaler = LoadModel(model_name, pair_type=pair_type, cv_folds=cv_folds)
        x_all_scaled = scaler.transform(x_all_pcainv)
        if model_name.split("_")[0] == "svm":
            dist_func = True
        else:
            dist_func = False
        
        scores = list()
        for rgs in estimators:
            scores.append(PrMapHelper(x_all_scaled, rgs, dist_func))
            # chunks = list()
            # for i in range(len(chunk_range)-1):
            #     chunks.append((x_all[chunk_range[i]:chunk_range[i+1]], rgs, dist_func))
            # pool = mp.Pool(processes=n_threads)
            # pool_out = pool.starmap(PrMapHelper, chunks)
            # pool.close()
            # scores.append(np.concatenate(pool_out))

        scores = np.stack(scores)
        scores = np.mean(scores, axis=0)
        scores = scores.reshape(xg[0].shape)
        scores -= np.amin(scores)
        scores /= np.amax(scores)
        scores_all.append(scores)

    # epsilon = 7.0 / 3 - 4.0 / 3 - 1
    # scores = - np.log(epsilon + scores / (1 - scores + epsilon))

    cm = plt.cm.RdBu
    model_titles = {"gb":"Gradient Boosting Regressor", "gb_pr":"Gradient Boosting Classifier",
                    "mlp":"Multi-layer Perceptron Regressor", "mlp_pr":"Multi-layer Perceptron Classifier",
                    "rf":"Random Forest Regressor", "rf_pr":"Random Forest Classifer",
                    "svm_rbf":"Support Vector Machine (RBF)", "svm_nys":"Support Vector Machine (Nystroem RBF)"}
    if single_model == None:
        fig, axs = plt.subplots(figsize=(22,12), nrows=4, ncols=2)
        ax_l = axs.reshape(-1)
        for i in range(len(models)):
            ax_l[i].contourf(xg[0], xg[1], scores_all[i], cmap=cm, alpha=0.8, levels=10,
                             vmin=np.amin(scores_all[i]), vmax=np.amax(scores_all[i]))
            # ax_l[i].imshow(scores_all[i], cmap=cm, alpha=0.8,
            #                vmin=np.amin(scores_all[i]), vmax=np.amax(scores_all[i]))
            if models[i][-4:] == "-ncv":
                ax_l[i].title.set_text(model_titles[models[i][:-4]])
            else:
                ax_l[i].title.set_text(model_titles[models[i]])
    else:
        plt.figure(figsize=(10,2))
        plt.contourf(xg[0], xg[1], scores_all[0], cmap=cm, alpha=0.8)
        if models[0][-4:] == "-ncv":
            plt.title(model_titles[models[0][:-4]])
        else:
            plt.title(model_titles[models[0]])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIDENT Train")
    parser.add_argument("-t", default=3, dest="threads", type=int, help="Number of threads (integer)")
    parser.add_argument("-m", default="train", dest="mode", choices=["train","train_cv","train_ncv","results","plot_map"], help="Program mode")
    parser.add_argument("-n", default="svm", dest="model_name", help="Model name")
    parser.add_argument("-s", default=0, dest="split", type=int, help="Split ID")
    parser.add_argument("-f", default=3, dest="folds", type=int, help="Number of CV folds")
    parser.add_argument("-k", default="rbf", dest="svm_kernel", help="SVM kernel type")
    parser.add_argument("-c", default=1.0, dest="svm_c", type=float, help="SVM C value")
    args = vars(parser.parse_args())

    mode = args["mode"]
    model_name = args["model_name"]

    if mode == "train":
        print("Starting non-cross-validated training...")
        PrTrain(model_name, pair_type="full")
    elif mode == "train_cv":
        print("Starting cross-validated training...")
        PrCrossVal(model_name, pair_type="full", n_folds=args["folds"], 
                   svm_kernel=args["svm_kernel"], svm_c=args["svm_c"])
    elif mode == "train_ncv":
        print("Starting nested cross-validated training...")
        NestedCV(model_name, pair_type="full", num_threads=args["threads"], 
                 split_id=args["split"], svm_kernel=args["svm_kernel"])
    elif mode == "results":
        PrCrossValResults(model_name)
    elif mode == "plot_map":
        PlotMap(n_threads=args["threads"])
    