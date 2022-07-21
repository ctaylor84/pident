import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd
import seaborn as sns

DAY_SECONDS = 86400
FIGURE_DPI = 150
SAVEFIG_DPI = 350
FIGURE_COLORS = ["red", "blue", "limegreen", "purple", "orange", 
                 "brown", "darkred", "darkblue", "green"]
MODEL_COLORS = {"gb":"red", "mlp":"blue", "rf":"limegreen", "svm_nys":"purple", "svm_rbf":"brown",
                "gb_pr":"darkred", "mlp_pr":"darkblue", "rf_pr":"green"}

def PlotRawSeries(group_mat, mode="scatter"):
    # mpl.rcParams["figure.dpi"] = FIGURE_DPI
    plt.figure(figsize=(12,4))

    animal_ids = np.unique(group_mat[:,2])
    pig_count = len(animal_ids)
    cm = plt.get_cmap("gist_ncar")
    rng = np.random.Generator(np.random.PCG64(890))
    colour_order = np.arange(pig_count)
    rng.shuffle(colour_order)
    colour_set = [cm(1.0*i/pig_count) for i in colour_order]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colour_set)

    for i in range(pig_count):
        true_indices = np.squeeze(np.argwhere(group_mat[:,2] == animal_ids[i]))
        true_series_i = group_mat[true_indices]
        if true_series_i.ndim == 1:
            true_series_i = np.expand_dims(true_series_i, 0)
        true_series_i = true_series_i[np.argsort(true_series_i[:,1])]
        if mode == "line":
            plt.plot(true_series_i[:,1] / DAY_SECONDS, true_series_i[:,0] / 1000)
        elif mode == "scatter":
            plt.scatter(true_series_i[:,1] / DAY_SECONDS, true_series_i[:,0] / 1000, s=10)
        else:
            raise RuntimeError("PlotRawSeries: Unknown mode: " + mode)
    
    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.show()

def PlotRawSeriesDists(group_mat, dist_mat, pred_labels):
    animal_ids = np.unique(group_mat[:,2])
    # mpl.rcParams["figure.dpi"] = FIGURE_DPI
    for animal_index in range(animal_ids.shape[0]):
        sum_vector = np.sum(dist_mat[pred_labels == animal_index, :], axis=0)
        fig, ax = plt.subplots(figsize=(12,8), nrows=2, ncols=1)
        ax[0].scatter(group_mat[:,1] / DAY_SECONDS, group_mat[:,0] / 1000, s=3, c=sum_vector, cmap="coolwarm")
        txt_1 = plt.text(0.02, 0.8, "A", fontsize=50, transform=ax[0].transAxes)
        txt_1.set_in_layout(False)
        ax[0].set_xlabel("Time (days)")
        ax[0].set_ylabel("Weight (Kg)")

        pred_series_i = group_mat[pred_labels == animal_index,:2]
        if pred_series_i.ndim == 1:
            pred_series_i = np.expand_dims(pred_series_i, 0)
        pred_series_i = pred_series_i[np.argsort(pred_series_i[:,1])]
        ax[1].plot(pred_series_i[:,1] / DAY_SECONDS, pred_series_i[:,0] / 1000, c="b")
        ax[1].scatter(group_mat[:,1] / DAY_SECONDS, group_mat[:,0] / 1000, s=3, c=sum_vector, cmap="coolwarm")
        txt_2 = plt.text(0.02, 0.8, "B", fontsize=50, transform=ax[1].transAxes)
        txt_2.set_in_layout(False)
        ax[1].set_xlabel("Time (days)")
        ax[1].set_ylabel("Weight (Kg)")
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        plt.tight_layout()
        plt.show()

def PlotTrajectories(group_mat, clustering_labels, n_clusters, pig_count):
    mpl.rcParams["figure.dpi"] = FIGURE_DPI
    animal_ids = np.unique(group_mat[:,2])
    true_series = list()
    pred_series = list()
    for i in range(n_clusters):
        pred_indices = np.squeeze(np.argwhere(clustering_labels == i))
        pred_series_i = group_mat[pred_indices,:2]
        if pred_series_i.ndim == 1:
            pred_series_i = np.expand_dims(pred_series_i, 0)
        pred_series.append(pred_series_i)

    for i in range(pig_count):
        true_indices = np.squeeze(np.argwhere(group_mat[:,2] == animal_ids[i]))
        true_series_i = group_mat[true_indices]
        if true_series_i.ndim == 1:
            true_series_i = np.expand_dims(true_series_i, 0)
        true_series.append(true_series_i)

    cm = plt.get_cmap("gist_ncar")
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[cm(1.0*i/pig_count) for i in range(pig_count)])
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,10))
    ax[0].title.set_text("True growth sequences")
    ax[1].title.set_text("Predicted growth sequences")
    for i in range(pig_count):
        ax[0].plot(true_series[i][:,1] / DAY_SECONDS, true_series[i][:,0] / 1000)
        ax[0].set_xlabel("Time (days)")
        ax[0].set_ylabel("Weight (Kg)")
    for i in range(n_clusters):
        pred_order = np.argsort(pred_series[i][:,1])
        ax[1].plot(pred_series[i][pred_order,1] / DAY_SECONDS, pred_series[i][pred_order,0] / 1000)
        ax[1].set_xlabel("Time (days)")
        ax[1].set_ylabel("Weight (Kg)")
    plt.tight_layout()
    plt.show()

def PlotComparison(true_to_pred, true_series, pred_series, plot_raw=False):
    pig_count = len(true_series)
    cm = plt.get_cmap("gist_ncar") # gist_rainbow, gist_ncar, hsv
    mpl.rcParams["figure.dpi"] = FIGURE_DPI
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[cm(1.0*i/pig_count) for i in range(pig_count)])
    plot_count = 3 if plot_raw else 2
    figsize = (12,12) if plot_raw else (12,10)
    fig, ax = plt.subplots(nrows=plot_count, ncols=1, figsize=figsize)
    #axes.set_color_cycle([cm(1.0*i/pig_count) for i in range(pig_count)])
    starting_weights = [true_series[i][0,0] for i in range(pig_count)]
    true_ids, pred_ids, starting_weights = zip(*sorted(zip(list(true_to_pred.keys()), 
                                                       list(true_to_pred.values()), starting_weights),
                                                       reverse=True, key=lambda x:x[2]))

    for true_id, pred_id in zip(true_ids, pred_ids):
        plot_index = 0
        if plot_raw:
            ax[plot_index].scatter(true_series[true_id][:,1] / DAY_SECONDS, 
                                   true_series[true_id][:,0] / 1000, s=1, c="r")
            ax[plot_index].title.set_text("True growth sequences (scatter)")
            ax[plot_index].set_xlabel("Time (days)")
            ax[plot_index].set_ylabel("Weight (Kg)")
            plot_index += 1
        ax[plot_index].plot(true_series[true_id][:,1] / DAY_SECONDS, true_series[true_id][:,0] / 1000)
        ax[plot_index].title.set_text("True growth sequences")
        ax[plot_index].set_xlabel("Time (days)")
        ax[plot_index].set_ylabel("Weight (Kg)")
        pred_order = np.argsort(pred_series[pred_id][:,1])
        plot_index += 1
        ax[plot_index].plot(pred_series[pred_id][pred_order,1] / DAY_SECONDS, 
                            pred_series[pred_id][pred_order,0] / 1000)
        ax[plot_index].title.set_text("Predicted growth sequences")
        ax[plot_index].set_xlabel("Time (days)")
        ax[plot_index].set_ylabel("Weight (Kg)")
    plt.tight_layout()
    plt.show()

def PlotComparisonCombo(true_to_pred, true_series, pred_series, save_fig=False):
    pig_count = len(true_series)
    cm = plt.get_cmap("gist_ncar") # gist_rainbow, gist_ncar, hsv
    mpl.rcParams["figure.dpi"] = FIGURE_DPI
    plt.figure(figsize=(12,6))
    #axes.set_color_cycle([cm(1.0*i/pig_count) for i in range(pig_count)])
    starting_weights = [true_series[i][0,0] for i in range(pig_count)]
    true_ids, pred_ids, starting_weights = zip(*sorted(zip(list(true_to_pred.keys()), 
                                                list(true_to_pred.values()), starting_weights),
                                                reverse=True, key=lambda x:x[2]))

    rng = np.random.Generator(np.random.PCG64(890))
    colour_order = np.arange(pig_count)
    rng.shuffle(colour_order)
    colour_set = [cm(1.0*i/pig_count) for i in colour_order]

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colour_set)
    for true_id in true_ids:
        plt.plot(true_series[true_id][:,1] / DAY_SECONDS, true_series[true_id][:,0] / 1000, linewidth=1.0)

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colour_set)
    for pred_id in pred_ids:
        pred_order = np.argsort(pred_series[pred_id][:,1])
        plt.plot(pred_series[pred_id][pred_order,1] / DAY_SECONDS, 
                 pred_series[pred_id][pred_order,0] / 1000, linestyle=":", linewidth=2.0)
    plt.xlabel("Time (days)")
    plt.ylabel("Weight (Kg)")
    plt.xlim((-1,36))
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    if save_fig:
        plt.savefig("plots/pident_comparison_plot_combo.jpg", dpi=SAVEFIG_DPI)
    else:
        plt.show()

def PlotTrajectoryTruths(true_to_pred, pred_series, true_mat, pred_mat):
    mpl.rcParams["figure.dpi"] = FIGURE_DPI
    tp = true_mat * pred_mat
    tn = (1-true_mat) * (1-pred_mat)
    fp = (1-true_mat) * pred_mat
    fn = true_mat * (1-pred_mat)
    cmap = mpl.colors.ListedColormap(["g", "r"])
    for i in range(len(pred_series)):
        plt.figure(figsize=(12,8))
        for series in pred_series:
            pred_order = np.argsort(series[:,1])
            plt.plot(series[pred_order,1] / DAY_SECONDS, series[pred_order,0] / 1000, c="gray")
        series_indices = np.squeeze(np.argwhere(pred_mat[:,i] == 1))
        pred_order = np.argsort(pred_series[true_to_pred[i]][:,1])
        colours = np.where(tp[series_indices,i] == 1, "g", "r")[pred_order]
        plt.plot(pred_series[true_to_pred[i]][pred_order,1] / DAY_SECONDS, 
                 pred_series[true_to_pred[i]][pred_order,0] / 1000, c="black")
        plt.scatter(pred_series[true_to_pred[i]][pred_order,1] / DAY_SECONDS, 
                    pred_series[true_to_pred[i]][pred_order,0] / 1000, c=colours)
        plt.xlabel("Time (days)")
        plt.ylabel("Weight (Kg)")
        plt.tight_layout()
        plt.show()

def PlotClusterMap(dist_mat, save_fig=False):
    # mpl.rcParams["figure.dpi"] = FIGURE_DPI
    #linkage_matrix = np.column_stack([clustering.children_, clustering.distances_])
    dist_array = ssd.squareform(dist_mat)
    dist_linkage = hierarchy.linkage(dist_array, method="complete")

    cmap = sns.cm.rocket_r
    sns.clustermap(dist_mat, row_linkage=dist_linkage, col_linkage=dist_linkage, cmap=cmap)
    if save_fig:
        plt.savefig("plots/pident_cluster_heatmap.png", dpi=500)
    else:
        plt.show()

def PlotDistScaling(group_mat, dist_mat, pig_count=10):
    mpl.rcParams["figure.dpi"] = FIGURE_DPI
    #embedding = MDS(n_components=500, n_jobs=4, dissimilarity="precomputed").fit_transform(dist_mat)
    #embedding = TSNE(n_components=2, n_jobs=4).fit_transform(embedding)
    embedding = TSNE(n_components=2, n_jobs=4, metric="precomputed").fit_transform(dist_mat)
    pig_ids = list()
    animal_ids = np.unique(group_mat[:,2])

    cm = plt.get_cmap("gist_ncar") # gist_rainbow, gist_ncar, hsv
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[cm(1.0*i/pig_count) for i in range(pig_count)])
    plt.figure(figsize=(12,8))

    for i in range(pig_count):
        indices = np.squeeze(np.argwhere(group_mat[:,2] == animal_ids[i]))
        plt.scatter(embedding[indices,0], embedding[indices,1], s=1)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()

def PlotBenchmarkComparison(mean_results, err_results, var_names, var_ranges, x_index=0, subplot_index=1):
    fig, ax = plt.subplots(figsize=(12,8), nrows=2, ncols=2)
    ax_l = ax.reshape(-1)
    series_length = 35

    model_index = 0
    for model_name, model_results in mean_results.items():
        model_errors = err_results[model_name]
        if model_name[-4:] == "-ncv":
            model_label = model_name[:-4]
        else:
            model_label = model_name
        for i, subplot_value in enumerate(var_ranges[subplot_index]):
            plot_values = list()
            plot_errors = list()
            for x_value in var_ranges[x_index]:
                result_code = str(x_value) + "-" + str(subplot_value).replace(".",",") + "-" + str(series_length)
                plot_values.append(model_results[result_code])
                plot_errors.append(model_errors[result_code])
            try:
                # ax_l[i].plot(var_ranges[x_index], plot_values, c=FIGURE_COLORS[model_index], label=model_name)
                ax_l[i].errorbar(var_ranges[x_index], plot_values, yerr=plot_errors, 
                                 c=MODEL_COLORS[model_label], label=model_label, capsize=5.0)
                ax_l[i].title.set_text(var_names[subplot_index] + ": " + str(subplot_value))
                ax_l[i].yaxis.grid(True)
            except IndexError:
                continue
        model_index += 1
    
    # for subplot in subplot_order[len(subplot_var_range):]:
    #     fig.delaxes(ax[subplot[0],subplot[1]])

    fig.text(0.48, 0.02, var_names[x_index], ha="center", va="center", fontsize="medium")
    fig.text(0.02, 0.5, "RMSE (Kg)", ha="center", va="center", rotation="vertical", fontsize="medium")
    handles, labels = ax_l[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=7, title="Model", fontsize="medium", frameon=False)
    plt.subplots_adjust(left=0.06, right=0.9, top=0.95, bottom=0.06)
    plt.show()
