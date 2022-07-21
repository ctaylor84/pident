import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.colors as pltc
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# pig_series column indices:
# 0 = sample ID
# 1 = pig ID
# 2 = location ID
# 3 = sample time (starts at zero)
# 4 = duration
# 5 = weight
# 6 = amount
# 7 = sample time (real time)

DAY_SECONDS = 86400
HOUR_SECONDS = 3600
SAMPLING_RATES_MIN = {"mean":0.7, "std":0.35, "min":0.2}
SAMPLING_RATES_MAX = {"mean":8.41, "std":2.98, "min":1.95}
SAMPLING_MAX_MULTI = 10.0
TIME_FREQS = [0.0572252, 0.05846027, 0.04322767, 0.06998765, 0.09468917, 0.22725401,
              0.29971182, 0.39069576, 0.49485385, 0.51132153, 0.52449568, 0.53561136,
              0.61259778, 0.74310416, 0.92095513, 1.0,        0.9691231,  0.74063401,
              0.45244957, 0.19596542, 0.09798271, 0.09880609, 0.10209963, 0.08892548]

POSITIVE_SAMPLINGS = 10
NEGATIVE_SAMPLINGS = 20
FIGURE_DPI = 150

class PriorityQueue():
    def __init__(self, num_groups):
        self.groups = list()
        self.group_ids = list()
        for i in range(num_groups):
            self.groups.append(list())
            self.group_ids.append(list())
        self.counts = [0] * num_groups
        self.sums = [0] * num_groups

    def _Sort(self):
        sorted_groups = zip(*sorted(zip(self.groups, self.group_ids, self.counts, self.sums), key=lambda x: (x[2], -x[3])))
        self.groups, self.group_ids, self.counts, self.sums_s = map(list, sorted_groups)

    def Insert(self, length, id):
        self.groups[0].append(length)
        self.group_ids[0].append(id)
        self.counts[0] += 1
        self.sums[0] += length
        self._Sort()

def GetFolds(pig_series, max_folds=10):
    lengths_dict = dict()
    for animal_id, animal_data in pig_series.items():
        lengths_dict[animal_id] = np.amax(animal_data[:,3]) / DAY_SECONDS
    fold_sets = dict()
    fold_sets_lengths = dict()
    sorted_ids, sorted_lengths = zip(*sorted(zip(lengths_dict.keys(), lengths_dict.values()), key=lambda x: x[1]))
    sorted_ids, sorted_lengths = list(sorted_ids), list(sorted_lengths)

    for bin_count in range(2, max_folds+1):
        priority_queue = PriorityQueue(bin_count)
        for i in range(len(sorted_ids)):
            priority_queue.Insert(sorted_lengths[i], sorted_ids[i])
        fold_sets[bin_count] = priority_queue.group_ids
        fold_sets_lengths[bin_count] = priority_queue.groups
    
    with open("data/data_folds.dat", "wb") as fh:
        pickle.dump(fold_sets, fh)

def DataSampling(animal_id, rng, pig_data, sample_rates, series_length, time_interp=False):
    trim_index = np.argmin(np.abs((pig_data[animal_id][:,3] / DAY_SECONDS) - series_length))
    weights = pig_data[animal_id][:trim_index,5]
    times = pig_data[animal_id][:trim_index,7]
    days_length = (times[-1] - times[0]) / DAY_SECONDS

    if time_interp:
        # Interpolation
        interp_times = np.arange(times[0], times[-1], step=HOUR_SECONDS)
        interp_weights = np.interp(interp_times, times, weights)

        # Select pig weights such that time frequency distribution matches the target
        start_hour = int(np.around((times[0] % DAY_SECONDS) / HOUR_SECONDS))
        time_freq_t = TIME_FREQS[start_hour:] + TIME_FREQS[:start_hour]
        time_freq_t = np.tile(time_freq_t, int(np.ceil(days_length)))[:interp_times.shape[0]]
        time_freq_r = rng.uniform(size=interp_times.shape)

        sample_indices = time_freq_t > time_freq_r
        sample_times = interp_times[sample_indices] - times[0]
        sample_weights = interp_weights[sample_indices]
    else:
        sample_times = pig_data[animal_id][:trim_index,3]
        sample_weights = weights

    # Match target sample frequency
    sampling_rate = rng.normal(loc=sample_rates["mean"], scale=sample_rates["std"])
    sampling_rate *= days_length
    sampling_rate = min(sampling_rate, sample_times.shape[0])
    sample_indices = np.arange(sample_times.shape[0])
    sample_count = max(sampling_rate, sample_rates["min"] * days_length, 3.0)
    sample_indices = rng.choice(sample_indices, size=round(sample_count), replace=False)
    sample_indices = np.sort(sample_indices)

    sample_times = sample_times[sample_indices]
    sample_weights = sample_weights[sample_indices]
    return sample_times, sample_weights

def DataSamplingUniform(rng, animal_id, pig_data, inc_mul=None, red_mul=0.25):
    weights = pig_data[animal_id][:,5]
    times = pig_data[animal_id][:,3]
    if inc_mul != None:
        interp_times = np.linspace(times[0], times[-1], num=int(np.ceil(weights.shape[0]*inc_mul)))
        weights = np.interp(interp_times, times, weights)
    sample_indices = rng.choice(weights.shape[0], size=int(weights.shape[0]*red_mul), replace=False)
    sample_times = times[sample_indices]
    sample_weights = weights[sample_indices]
    return sample_times, sample_weights

def GetGroupPairs(group_matrix):
    group_weights = group_matrix[:,0]
    group_times = group_matrix[:,1]
    group_ids = group_matrix[:,2]
    group_indices = np.expand_dims(np.arange(group_weights.shape[0]), 1)

    pair_weights = np.transpose(np.meshgrid(group_weights, group_weights)).reshape(-1,2)
    pair_times = np.transpose(np.meshgrid(group_times, group_times)).reshape(-1,2)
    pair_times = np.expand_dims(pair_times[:,1] - pair_times[:,0], 1)

    pair_ids = np.transpose(np.meshgrid(group_ids, group_ids)).reshape(-1,2)
    pos_ids = np.argwhere(pair_ids[:,0] == pair_ids[:,1])
    pair_cat = np.expand_dims(np.repeat([0], pair_weights.shape[0]), 1)
    pair_cat[pos_ids] = 1

    pair_indices = np.transpose(np.meshgrid(group_indices, group_indices)).reshape(-1,2)
    #pair_times_all = np.transpose(np.meshgrid(group_times, group_times)).reshape(-1,2)
    group_pairs = np.concatenate((pair_weights, pair_times, pair_cat, pair_indices), axis=1)

    # remove backward pairs (negative time delta)
    group_pairs = group_pairs[group_pairs[:,2] > 0.0]
    return group_pairs

def UpdateSampleRates(sample_rate):
    sample_rate = min(sample_rate, SAMPLING_MAX_MULTI)
    stat_names = ["mean", "min", "std"]
    new_rates = dict()
    for stat in stat_names:
        new_rates[stat] = SAMPLING_RATES_MIN[stat] * sample_rate
    return new_rates

def GroupSampling(pig_data, group_id=0, pig_count=55, series_length=35, 
                  sample_rate=1.0, fold_set=None):
    rng = np.random.Generator(np.random.PCG64(group_id))
    if fold_set != None:
        all_ids = fold_set
    else:
        all_ids = pig_data.keys()
    id_set = list()
    for animal_id in all_ids:
        if pig_data[animal_id][-1,3] / DAY_SECONDS > series_length:
            id_set.append(animal_id)
    animal_ids = rng.choice(id_set, size=pig_count, replace=False)
    sample_rates = UpdateSampleRates(sample_rate)

    group_weights = list()
    group_times = list()
    group_ids = list()
    for i in animal_ids:
        group_times_i, group_weights_i = DataSampling(i, rng, pig_data, sample_rates, series_length)
        group_weights.append(group_weights_i)
        group_times.append(group_times_i)
        group_ids.append(np.repeat([i], group_times[-1].shape[0]))
    group_weights = np.concatenate(group_weights)
    group_times = np.concatenate(group_times)
    group_ids = np.concatenate(group_ids)
    group_mat = np.stack((group_weights, group_times, group_ids), axis=1)
    group_mat = group_mat[np.argsort(group_mat[:,1])] # sort by time
    return group_mat

def PairGen(pig_data, fold_sets, fold_count=3, pair_type="full"):
    rng = np.random.Generator(np.random.PCG64(999))
    fold_set = fold_sets[fold_count]
    fold_pairs = list()
    class_counts = list()
    pos_count_all = 0
    neg_count_all = 0

    for fold_n in range(fold_count):
        animal_ids = fold_set[fold_n]
        pairs = list()
        total_pigs = len(animal_ids)
        current_pig = 0
        for animal_id in animal_ids:
            current_pig += 1
            print(current_pig, "/", total_pigs)
            id_set = set(animal_ids)
            id_set.remove(animal_id)
            id_set = list(id_set)

            # self_times, self_weights = DataSamplingUniform(rng, animal_id, pig_data, 
            #                                                inc_mul=POSITIVE_SAMPLINGS, red_mul=1)
            self_weights = pig_data[animal_id][:,5]
            self_times = pig_data[animal_id][:,3]
            pos_weights = np.transpose(np.meshgrid(self_weights, self_weights)).reshape(-1,2)
            pos_times = np.transpose(np.meshgrid(self_times, self_times)).reshape(-1,2)
            pos_times = np.expand_dims(pos_times[:,1] - pos_times[:,0], 1)

            # restrict "next time point" to the actual next time point (and no following from the same pig)
            if pair_type == "single":
                pos_indices = np.arange(0, self_times.shape[0], dtype=int)
                pos_indices = np.transpose(np.meshgrid(pos_indices, pos_indices)).reshape(-1,2)
                pos_indices = np.argwhere(pos_indices[:,1] - pos_indices[:,0] == 1)
                pos_cat = np.expand_dims(np.repeat([0], pos_weights.shape[0]), 1)
                pos_cat[pos_indices] = 1 # all remaining "non next" point is a negative sample
            elif pair_type == "full":
                pos_cat = np.expand_dims(np.repeat([1], pos_weights.shape[0]), 1)
            pos_pairs = np.concatenate((pos_weights, pos_times, pos_cat), axis=1)

            self_times_f, self_weights_f = DataSamplingUniform(rng, animal_id, pig_data)
            target_weights = list()
            target_times = list()
            target_negative_indices = rng.choice(id_set, size=NEGATIVE_SAMPLINGS, replace=False)
            for i in target_negative_indices:
                target_times_i, target_weights_i = DataSamplingUniform(rng, i, pig_data)
                target_weights.append(target_weights_i)
                target_times.append(target_times_i)
            target_weights = np.concatenate(target_weights)
            target_times = np.concatenate(target_times)

            neg_weights = np.transpose(np.meshgrid(self_weights_f, target_weights)).reshape(-1,2)
            neg_times = np.transpose(np.meshgrid(self_times_f, target_times)).reshape(-1,2)
            neg_times = np.expand_dims(neg_times[:,1] - neg_times[:,0], 1)
            neg_cat = np.expand_dims(np.repeat([0], neg_weights.shape[0]), 1)

            neg_full = np.concatenate((neg_weights, neg_times, neg_cat), axis=1)
            full_pairs = np.concatenate((pos_pairs, neg_full), axis=0)

            # remove backward pairs (negative time delta)
            full_pairs = full_pairs[full_pairs[:,2] > 0.0]

            pairs.append(full_pairs)
        pairs = np.concatenate(pairs, axis=0)

        pos_indices = np.squeeze(np.argwhere(np.around(pairs[:,3]) == 1))
        neg_indices = np.squeeze(np.argwhere(np.around(pairs[:,3]) == 0))
        pos_indices = rng.choice(pos_indices, size=int(pos_indices.shape[0] * 0.025), replace=False)
        neg_indices = rng.choice(neg_indices, size=int(neg_indices.shape[0] * 0.025), replace=False)
        all_indices = np.concatenate((pos_indices, neg_indices))
        rng.shuffle(all_indices)
        pairs = pairs[all_indices,:]
        fold_pairs.append(pairs)

        positive_count = sum(pairs[:,3])
        negative_count = sum(pairs[:,3] - 1) * -1
        pos_count_all += positive_count
        neg_count_all += negative_count
        class_counts.append([total_pigs, positive_count, negative_count, 100 * positive_count / (positive_count + negative_count)])
    
    for i, counts in enumerate(class_counts):
        print("Fold ", str(i+1) + ":", counts)
    print("\nTotal:", pos_count_all, neg_count_all, 100 * pos_count_all / (pos_count_all + neg_count_all))
    with open("data/data_pairs_" + str(fold_count) + "fold_" + pair_type + ".dat", "wb") as fh:
        pickle.dump(fold_pairs, fh)

def SeqRoll(seq, window):
    shape = (seq.size - window + 1, window)
    strides = (seq.itemsize, seq.itemsize)
    return np.lib.stride_tricks.as_strided(seq, shape=shape, strides=strides)

def RegressionProcess(process_id, data_sets, pairs, scaler, indices, out_queue):
    regression_scores = list()
    for i in indices:
        if i % 1000 == 0:
            print(round(i / pairs.shape[0], 0) * 100, "%")
        combined_pairs = np.concatenate((data_sets[pairs[i,0]], data_sets[pairs[i,1]]), axis=1).T
        combined_pairs = scaler.transform(combined_pairs)
        pair_x = combined_pairs[:,1].reshape(-1,1)
        pair_y = combined_pairs[:,0].reshape(-1,1)
        reg = LinearRegression(n_jobs=1).fit(pair_x, pair_y)
        regression_scores.append(np.asarray([reg.score(pair_x, pair_y), reg.coef_[0]]))
    regression_scores = np.stack(regression_scores)
    out_queue.put((process_id, regression_scores))

def PairExploration(pair_type="full"):
    if pair_type == "full":
        pairs = np.load("data/data_pairs_full.npy")
    elif pair_type == "single":
        pairs = np.load("data/data_pairs_single.npy")

    #sample_indices = np.random.choice(pairs.shape[0], size=500000, replace=False)
    x_all = np.stack((pairs[:,0], pairs[:,2],
                     (pairs[:,1] - pairs[:,0]) / pairs[:,2],
                     np.abs(pairs[:,1] - pairs[:,0]),
                     pairs[:,1] - pairs[:,0]), axis=1)
    #x_all = StandardScaler().fit_transform(x_all)
    x_all = x_all[:,:]
    y_all = pairs[:,3].astype(int)

    colours = ["blue","red"]
    mpl.rcParams["figure.dpi"] = FIGURE_DPI
    # plt.figure(figsize=(16,12))
    # plt.scatter(x_all[:,0], x_all[:,2], c=y_all, s=1, cmap=pltc.ListedColormap(colours))
    # ax = plt.gca()
    # ax.set_ylim([-0.5,0.5])
    # plt.show()
    
    var_names = dict({0:"Initial weight (g)",
                      1:"Time difference (s)",
                      2:"weight difference / time difference (g/s)",
                      3:"abs(final weight - initial weight) (g)",
                      4:"weight difference (g)"})
    var_limits = dict({0:-1, 1:-1, 2:50, 3:-1, 4:-1})
    var_combins = [(0,3),(1,3),(0,4),(1,4),(0,2)]
    fig, ax = plt.subplots(nrows=1, ncols=len(var_combins), figsize=(24,15))
    for i in range(len(var_combins)):
        var_pair = var_combins[i]
        ax[i].scatter(x_all[:,var_pair[0]], x_all[:,var_pair[1]], c=y_all, s=1, cmap=pltc.ListedColormap(colours))
        ax[i].set_xlabel(var_names[var_pair[0]])
        ax[i].set_ylabel(var_names[var_pair[1]])
        if var_limits[var_pair[1]] != -1:
            ax[i].set_ylim([0,var_limits[var_pair[1]]])
    plt.tight_layout()
    plt.show()

    positive_count = sum(y_all)
    negative_count = sum(y_all - 1) * -1
    print(positive_count, negative_count, negative_count / positive_count, 100 * positive_count / (positive_count + negative_count))

if __name__ == "__main__":
    with open("data/data_dump.dat", "rb") as fh:
        pig_data = pickle.load(fh)
    # GetFolds(pig_data["pig_series"])
    with open("data/data_folds.dat", "rb") as fh:
        fold_sets = pickle.load(fh)
    PairGen(pig_data["pig_series"], fold_sets, fold_count=3)
    # PairExploration()
    # PairGen(pig_data["pig_series"], pair_type="single")
    # PairExploration(pair_type="single")
    # DataSampling(list(pig_data["pig_series"].keys())[0], pig_data["pig_series"])
