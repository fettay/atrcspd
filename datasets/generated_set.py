import numpy as np
import pandas as pd
from tqdm import tqdm
np.random.seed(26)


EXAMPLES_BY_CATEGORY = 20


def random_except(start, stop, except_nb):
    return np.random.choice(list(range(start, except_nb)) + list(range(except_nb+1, stop)))


def generate_gaussian_ts(max_mean=10, max_std=10, ts_size=3744):
    """
    Create changes in gaussian ts
    """
    first_mean, first_std = np.random.randint(0, max_mean), np.random.randint(1, max_mean)
    mean_or_var = np.random.randint(0, 3)
    if mean_or_var == 0:
        second_mean, second_std = first_mean, random_except(1, 10, first_std)
    elif mean_or_var == 1:
        second_mean, second_std = random_except(1, 10, first_mean), first_std
    else:
        second_mean, second_std = random_except(1, 10, first_mean), random_except(1, 10, first_std)
    ts = np.concatenate([np.random.normal(first_mean, first_std, size=int(ts_size/1.5)),
                          np.random.normal(second_mean, second_std, size=int(ts_size/4))])
    return ts


def add_spikes(ts, spike_length = None, spike_height=None, periodic=False, spike_freq=None, noise_factor=5, get_spikes=False):
    
    ts = np.copy(ts)
    spike_max_length = 15
    spike_max_height = 10
    if spike_height is None:
        spike_height = np.random.randint(1, spike_max_height) * np.max(ts)
    if spike_length is None:
        spike_length = np.random.randint(5, spike_max_length)
    spike_cut = np.random.randint(2, spike_length-1)
    if spike_freq is None:
        spike_freq = np.random.randint(1, 4 * int(len(ts) / spike_length ))
    else:
        spike_freq = min(spike_freq, 8 * int(len(ts) / spike_length ))

    def create_spike(index):
        start = ts[index]
        spike = np.ones(spike_length) * spike_height + np.random.normal(0, spike_height/noise_factor, size=spike_length) + start
        return spike
        

    if periodic:
        start = np.random.randint(0, spike_freq)
        pos = np.arange(start=start, stop=len(ts) - spike_length - 1, step=spike_freq)

    else:
        _, pos = np.where([np.random.choice([True, False], p=[1/spike_freq, 1-(1/spike_freq)], size=len(ts) - spike_length - 1)])
    
    all_spikes = np.array([create_spike(p) for p in pos]).reshape(-1)
    
    old_pos = list(pos)
    for p in old_pos:
        pos = np.concatenate((pos, [p + i for i in range(1, spike_length)]))

    ts[pos] = all_spikes
    if get_spikes:
        return ts, pos
    return ts
        


def generate_seasonal_ts(size=3744, noise_factor=20, val_1=None, val_2=None, length_1=None, length_2=None, freq=288):
    if length_1 is None:
        length_1 = np.random.randint(0, freq-20)
    if length_2 is None:
        length_2 = np.random.randint(0, freq - length_1)
    if val_1 is None:
        val_1 = np.random.randint(900)
    if val_2 is None:
        val_2 = np.random.randint(val_1 + 100, 1000)
    def get_pattern():
        pattern = np.concatenate([np.random.normal(loc=val_1, scale=val_1/20, size=length_1),
                                  np.random.normal(loc=val_2, scale=val_2/noise_factor, size=length_2), 
                                  np.random.normal(loc=val_1, scale=val_1/noise_factor, size=freq - length_1 - length_2)], axis=0)
        return pattern
    return np.concatenate([get_pattern() for _ in range(size // freq)])




def get_gaussian_samples(size=3744):
    means = np.random.randint(0, 1000, size=EXAMPLES_BY_CATEGORY)
    stds = np.random.randint(2, 20, size=EXAMPLES_BY_CATEGORY)
    stds = means / stds
    return np.array([np.random.normal(m, s, size=size) for m, s in zip(means, stds)])

def generate_set():

    all_ts, all_labels = [], []

    # Changes in gaussian, with and without random spikes  0-20, 20-40
    gaussian_changes = ([generate_gaussian_ts() for _ in range(EXAMPLES_BY_CATEGORY)], [(2400,) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(gaussian_changes[0])
    all_labels.extend(gaussian_changes[1])
    gaussian_changes_spikes = ([add_spikes(generate_gaussian_ts(), noise_factor=np.random.randint(1, 10)) for _ in range(EXAMPLES_BY_CATEGORY)], [(2400,) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(gaussian_changes_spikes[0])
    all_labels.extend(gaussian_changes_spikes[1])


    # Generate gaussian TS with change in spike frequency 40-60
    frequency_1 = np.random.randint(150, 400, size=EXAMPLES_BY_CATEGORY)
    frequency_2 = np.random.randint(20, 50, size=EXAMPLES_BY_CATEGORY)
    gaussian_samples = get_gaussian_samples()
    spike_height = np.random.randint(1, 10, size=EXAMPLES_BY_CATEGORY) * np.max(gaussian_samples)
    spike_length = np.random.randint(10, 30, size=EXAMPLES_BY_CATEGORY)
    gaussian_spike_changes_freq = [np.concatenate([add_spikes(g[:2600], spike_freq=f1, spike_height=sh, spike_length=sl), add_spikes(g[2600:], spike_freq=f2, spike_height=sh, spike_length=sl)]) 
                                    for g, f1, f2, sh, sl in zip(gaussian_samples, frequency_1, frequency_2, spike_height, spike_length)]
    gaussian_spike_changes_freq = (gaussian_spike_changes_freq, [(2600, ) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(gaussian_spike_changes_freq[0])
    all_labels.extend(gaussian_spike_changes_freq[1])

    # Generate gaussian TS with change in spike height 60-80
    gaussian_samples = get_gaussian_samples()
    height_1 = np.random.uniform(1, 9, size=EXAMPLES_BY_CATEGORY) * np.max(gaussian_samples)
    height_2 = np.random.uniform(1.5, 3, size=EXAMPLES_BY_CATEGORY) * height_1
    spike_freq = np.random.randint(20, 100, size=EXAMPLES_BY_CATEGORY)
    spike_length = np.random.randint(10, 30, size=EXAMPLES_BY_CATEGORY)
    gaussian_spike_changes_height = [np.concatenate([add_spikes(g[:2600], spike_freq=sf, spike_height=h1, spike_length=sl), add_spikes(g[2600:], spike_freq=sf, spike_height=h2, spike_length=sl)]) 
                                    for g, h1, h2, sf, sl in zip(gaussian_samples, height_1, height_2, spike_freq, spike_length)]
    gaussian_spike_changes_height = (gaussian_spike_changes_height, [(2600, ) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(gaussian_spike_changes_height[0])
    all_labels.extend(gaussian_spike_changes_height[1])

    # Generate gaussian TS with change height in seasonal spikes 80-100
    gaussian_samples = get_gaussian_samples()
    height_1 = np.random.uniform(1, 9, size=EXAMPLES_BY_CATEGORY) * np.max(gaussian_samples)
    height_2 = np.random.uniform(1.5, 3, size=EXAMPLES_BY_CATEGORY)
    spike_freq = np.random.randint(288, 289, size=EXAMPLES_BY_CATEGORY)
    spike_length = np.random.randint(10, 30, size=EXAMPLES_BY_CATEGORY)
    gaussian_spike_periodic = [add_spikes(g, spike_freq=sf, spike_height=h1, spike_length=sl, periodic=True, get_spikes=True) for g, h1, sf, sl in zip(gaussian_samples, height_1, spike_freq, spike_length)]
    good_ts = []
    for ts_spikes, h2 in zip(gaussian_spike_periodic, height_2):
        ts, spikes = ts_spikes
        ts[spikes[spikes>=2600]] = ts[spikes[spikes>=2600]] * h2
        good_ts.append(ts)

    gaussian_spike_periodic = (good_ts, [(2600, ) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(gaussian_spike_periodic[0])
    all_labels.extend(gaussian_spike_periodic[1])


    # Generate gaussian TS with change time in seasonal spike 100-120
    gaussian_samples = get_gaussian_samples()
    spike_height = np.random.randint(1, 10, size=EXAMPLES_BY_CATEGORY) * np.max(gaussian_samples)
    spike_length = np.random.randint(10, 30, size=EXAMPLES_BY_CATEGORY)
    gaussian_spike_changes_pos = [np.concatenate([add_spikes(g[:2600], spike_freq=288, spike_height=sh, spike_length=sl, periodic=True), add_spikes(g[2600:], spike_freq=288, spike_height=sh, spike_length=sl, periodic=True)]) 
                                    for g, sh, sl in zip(gaussian_samples, spike_height, spike_length)]
    gaussian_spike_changes_pos = (gaussian_spike_changes_pos, [(2600, ) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(gaussian_spike_changes_pos[0])
    all_labels.extend(gaussian_spike_changes_pos[1])


    # Generate periodic TS with change in period height  120-140
    height_1 = np.random.randint(900, size=EXAMPLES_BY_CATEGORY)
    height_2 = np.array([np.random.randint(h1 + 100, 1000) for h1 in height_1])
    height_3 = np.array([np.random.randint(h2 + 100, 1300) for h2 in height_2])
    length_1 = np.random.randint(0, 288, size=EXAMPLES_BY_CATEGORY)
    length_2 = np.array([np.random.randint(0, 288 - l1) for l1 in length_1])
    noise_factors = np.random.randint(5, 30, size=EXAMPLES_BY_CATEGORY)
    periodic1 = [generate_seasonal_ts(val_1=h1, val_2=h2, length_1=l1, length_2=l2, noise_factor=nf) for h1, h2, l1, l2, nf in zip(height_1, height_2, length_1, length_2, noise_factors)]
    periodic2 = [generate_seasonal_ts(val_1=h1, val_2=h3, length_1=l1, length_2=l2, noise_factor=nf) for h1, h3, l1, l2, nf in zip(height_1, height_3, length_1, length_2, noise_factors)]
    new_data = np.array([np.concatenate([p1[:2400], p2[2400:]])  for p1, p2 in zip(periodic1, periodic2)])
    # new_data[:int(EXAMPLES_BY_CATEGORY/2)] = np.array([add_spikes(p, spike_freq=np.random.randint(400, 1000)) for p in new_data[:int(EXAMPLES_BY_CATEGORY/2)]])  # Add spikes to half
    periodic_changes_height = (new_data, [(2400, ) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(periodic_changes_height[0])
    all_labels.extend(periodic_changes_height[1])



    # Generate periodic TS with change in resting period height  140-160
    height_1 = np.random.randint(900, size=EXAMPLES_BY_CATEGORY)
    height_2 = np.array([np.random.randint(h1 + 100, 1000) for h1 in height_1])
    height_3 = np.array([np.random.randint(h2 + 100, 1300) for h2 in height_2])
    length_1 = np.random.randint(0, 288, size=EXAMPLES_BY_CATEGORY)
    length_2 = np.array([np.random.randint(0, 288 - l1) for l1 in length_1])
    noise_factors = np.random.randint(5, 30, size=EXAMPLES_BY_CATEGORY)
    periodic1 = [generate_seasonal_ts(val_1=h1, val_2=h3, length_1=l1, length_2=l2, noise_factor=nf) for h1, h3, l1, l2, nf in zip(height_1, height_3, length_1, length_2, noise_factors)]
    periodic2 = [generate_seasonal_ts(val_1=h2, val_2=h3, length_1=l1, length_2=l2, noise_factor=nf) for h2, h3, l1, l2, nf in zip(height_2, height_3, length_1, length_2, noise_factors)]
    new_data = np.array([np.concatenate([p1[:2400], p2[2400:]])  for p1, p2 in zip(periodic1, periodic2)])
    # new_data[:int(EXAMPLES_BY_CATEGORY/2)] = np.array([add_spikes(p, spike_freq=np.random.randint(400, 1000)) for p in new_data[:int(EXAMPLES_BY_CATEGORY/2)]])  # Add spikes to half
    periodic_changes_resting_height = (new_data, [(2400, ) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(periodic_changes_resting_height[0])
    all_labels.extend(periodic_changes_resting_height[1])


    # Generate periodic TS with change in period length 160-180
    height_1 = np.random.randint(900, size=EXAMPLES_BY_CATEGORY)
    height_2 = np.array([np.random.randint(h1 + 100, 1000) for h1 in height_1])
    length_1 = np.random.randint(0, 200, size=EXAMPLES_BY_CATEGORY)
    length_2 = np.array([np.random.randint(0, 200 - l1) for l1 in length_1])
    length_3 = np.array([np.random.randint(l2, 288 - l1) for l1, l2 in zip(length_1, length_2)])
    noise_factors = np.random.randint(5, 30, size=EXAMPLES_BY_CATEGORY)
    periodic1 = [generate_seasonal_ts(val_1=h1, val_2=h2, length_1=l1, length_2=l2, noise_factor=nf) for h1, h2, l1, l2, nf in zip(height_1, height_2, length_1, length_2, noise_factors)]
    periodic2 = [generate_seasonal_ts(val_1=h1, val_2=h2, length_1=l1, length_2=l3, noise_factor=nf) for h1, h2, l1, l3, nf in zip(height_1, height_2, length_1, length_3, noise_factors)]
    new_data = np.array([np.concatenate([p1[:2400], p2[2400:]])  for p1, p2 in zip(periodic1, periodic2)])
    # new_data[:int(EXAMPLES_BY_CATEGORY/2)] = np.array([add_spikes(p, spike_freq=np.random.randint(400, 1000)) for p in new_data[:int(EXAMPLES_BY_CATEGORY/2)]])  # Add spikes to half
    periodic_changes_duration = (new_data, [(2400, ) for _ in range(EXAMPLES_BY_CATEGORY)])
    all_ts.extend(periodic_changes_duration[0])
    all_labels.extend(periodic_changes_duration[1])



    ###
    # NEGATIVES
    ###
    regular_gaussians = np.concatenate([get_gaussian_samples(), get_gaussian_samples()])
    gaussian_samples_spikes = [add_spikes(s) for s in np.concatenate([get_gaussian_samples(), get_gaussian_samples()])]
    gaussian_samples_periodic_spikes = [add_spikes(s, periodic=True, spike_freq=288) for s in np.concatenate([get_gaussian_samples(), get_gaussian_samples()])]
    periodic = [generate_seasonal_ts() for _ in range(EXAMPLES_BY_CATEGORY * 2)]
    all_ts.extend(regular_gaussians)
    # all_ts.extend(gaussian_samples_spikes)
    all_ts.extend(gaussian_samples_periodic_spikes)
    all_ts.extend(periodic)
    all_labels.extend([tuple() for _ in range(40*3)])


    all_ts = [pd.Series(data=ts, index=pd.date_range(start="2018-02-22", periods=len(ts), freq="5T")) for ts in all_ts]
    
    return all_ts, all_labels


def get_result(pred, label):
    if len(label) == 0:
        if pred is None:
            return True
        return False
    else:
        if pred is None:
            return False
        if np.abs(pred[0]-label[0]) < 288:
            return True
        return False


def compute_model(model, all_ts, labels):
    results = []
    for ts, label in tqdm(zip(all_ts, labels), total=len(all_ts)):
        pred = model(ts)[0]
        results.append(get_result(pred, label))
    return np.array(results)


def get_detection_by_group(res, has_ano):
    print("Detection on Category A")
    print(res[has_ano==True][:20].sum() / 20)
    print("Detection on Category B")
    print(res[has_ano==True][20:40].sum() / 20)
    # print("Detection with change in spike freq")
    # print(res[has_ano==True][40:60].sum() / 20)
    # print("Detection with change in random spike height")
    # print(res[has_ano==True][60:80].sum() / 20)
    print("Detection on Category C")
    print(res[has_ano==True][80:100].sum() / 20)
    print("Detection on Category D")
    print(res[has_ano==True][100:120].sum() / 20)
    print("Detection on Category E")
    print(res[has_ano==True][120:140].sum() / 20)
    print("Detection on Category F")
    print(res[has_ano==True][140:160].sum() / 20)
    print("Detection on Category G")
    print(res[has_ano==True][160:180].sum() / 20)
    print("False Positive Rate")
    print((~res[has_ano==False]).sum() / res[has_ano==False].shape[0])


def run_generated_set(model):
    """Run a model function on the generated set
    
    Arguments:
        model {[type]} -- [description]
    """
    all_ts, all_labels = generate_set()
    results = compute_model(model, all_ts, all_labels)
    has_ano = np.array([len(l) > 0 for l in all_labels])
    get_detection_by_group(results, has_ano)
    return results
