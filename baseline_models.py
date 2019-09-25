import numpy as np
from scipy.signal import find_peaks


def kcpe(ts, pen=10, window_size=288*2, min_size=24*6, jump=5):
    from ruptures import Window

    ts_samples = ts
    model = Window(model='rbf', min_size=min_size, width=window_size, jump=jump)
    model.fit(ts.values)
    results = model.predict(pen=pen)[:-1]
    if len(results) == 0:
        return (None, None)
    return (results, None)


def rulsif(ts, window_size=50, threshold=.1):
    from densratio import densratio

    def make_window(arr, win, jump):
        return np.array([arr[i:i+win] for i in range(0, len(arr)-win+1, jump)])

    ts = (ts - ts.min(axis=0)) / (ts.max(axis=0) - ts.min(axis=0))
    all_ratios = []
    rolling_window = np.array(make_window(ts, window_size, window_size))
    for win1, win2 in zip(rolling_window[:-1], rolling_window[1:]):
        concat = np.concatenate([win1, win2])
        med = np.nanmedian(concat)
        sigma_list = med * np.array([.6,.8,1.0,1.2,1.4])
        lambda_list = [10e-3, 10e-2, 10e-1, 1, 10]
        ratio = densratio(win1, win2, alpha=0.01, lambda_range=lambda_list, sigma_range=sigma_list, verbose=False)
        all_ratios.append(ratio.alpha_PE)

    preds = _find_bp_rulsif(ts, all_ratios, threshold, window_size)
    return (preds.tolist(), None)


def _find_bp_rulsif(ts, scores, thresh, win_size):
    peaks, _ = find_peaks(scores)
    scores_peaks = {p: scores[p] for p in peaks}
    idx = np.array([p for p, v in scores_peaks.items() if v > 0.05])
    if len(idx) == 0:
        return None
    return (idx + 1)*win_size


def run_kcpe_wrapper(pen=10, window_size=288*2, min_size=24*6, jump=5):
    def run_kcpe(ts):
        return kcpe(ts, pen=pen, window_size=window_size, min_size=min_size, jump=jump)
    return run_kcpe


def run_rulsif_wrapper(window_size=50, threshold=.1):
    def run_rulsif(ts):
        return rulsif(ts, window_size=window_size, threshold=threshold)
    return run_rulsif