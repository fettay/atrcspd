"""
Detects baseline change using Autoencoder
"""

from torch import nn
import torch
import numpy as np


EPOCHS = 100
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-5
DISTANCE_THRESHOLD = 1e-1
PENALTY_COEFFICIENT = 5e-3
MIN_WINDOW_SIZE = 16
MIN_SILHOUETTE_SCORE = .7
DISTANCE_METRIC = 'cosine'



class _Autoencoder(nn.Module):
    """
    Defines the network architecture with pytorch
    """

    def __init__(self, window_size):
        super().__init__()
        self.hidden_dim = window_size // 16
        self.reduce1 = nn.Linear(window_size, self.hidden_dim*4, bias=True)
        self.reduce2 = nn.Linear( self.hidden_dim*4, self.hidden_dim*2, bias=True)
        self.reduce3 = nn.Linear(self.hidden_dim*2, self.hidden_dim, bias=True)
        self.increase1 = nn.Linear(self.hidden_dim, self.hidden_dim*2, bias=True)
        self.increase2 = nn.Linear(self.hidden_dim*2, window_size)

    def forward(self, window):
        reduce1 = torch.tanh(self.reduce1(window.view(len(window), 1, -1)))
        reduce2 = torch.tanh(self.reduce2(reduce1))
        reduce3 = self.reduce3(reduce2)
        pred =  torch.tanh(self.increase2(torch.tanh(self.increase1(reduce3))))
        return pred, reduce3


class ATR():
    """ Wrapper for the baseline change algorithm using autoencoder
    """

    def __init__(self, penalty_coefficient: float=PENALTY_COEFFICIENT, weight_decay: float=WEIGHT_DECAY,
        distance_threshold: float=DISTANCE_THRESHOLD, min_silhouette_score: float=MIN_SILHOUETTE_SCORE,
        learning_rate: float=LEARNING_RATE, epochs: int=EPOCHS):
        """
        Keyword Arguments:
            penalty_coefficient {float} -- L1 penalty coefficient for difference between embedding of consecutive windows (see function _custom_loss) (default: {PENALTY_COEFFICIENT})
            weight_decay {float} -- L2 penalty for the weights of the model (default: {WEIGHT_DECAY})
            distance_threshold {float} -- Minimum distance between two clusters to allow a split (default: {DISTANCE_THRESHOLD})
            min_silhouette_score {float} -- Minimum silhouette score to create clusters
            learning_rate {float} -- Learning rate for the autoencoder (default: {LEARNING_RATE})
            epochs {int} -- Number of training epochs to use (default: {EPOCHS})
        """
        self.penalty_coefficient = penalty_coefficient
        self.weight_decay = weight_decay
        self.distance_threshold = distance_threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_silhouette_score = min_silhouette_score

    @staticmethod
    def _make_window(arr, win):
        """
        Rolling window from an array
        """
        return np.array([arr[i:i+win] for i in range(0, len(arr)-win+1, win)])

    @staticmethod
    def _ts_to_input(ts, window_size):
        """
        Takes a TimeSeries object and tranform it to torch inputs and outputs
        """
        inputs = torch.tensor(ATR._make_window(ts, window_size)).reshape(-1, 1, window_size).float()
        outputs = torch.tensor(ATR._make_window(ts, window_size)).reshape(-1, 1, window_size).float()
        return inputs, outputs

    @staticmethod
    def _scale_ts(ts):
        """
        Move the TimeSeries to a 0, 1 range
        """
        return (ts - ts.min(axis=0)) / (ts.max(axis=0) - ts.min(axis=0))

    @staticmethod
    def _get_window_size(ts, period):
        """
        Computes window size from seasonality and ts size
        """
        if period < MIN_WINDOW_SIZE:  # Concatenate several windows
            return (MIN_WINDOW_SIZE // period + 1) * period

        return period

    def _get_penalty_factor(self, windows_count):
        """
        Computes the penalty factor from the ts size and the window size (number of windows) 
        """
        return (1/windows_count)*self.penalty_coefficient

    @staticmethod
    def _custom_loss(pred, outputs, hidden, penalty_factor):
        """
        Compute the loss by taking MSE + sum |x_{n+1} - x_n| * penalty_factor
        """
        mse = nn.functional.mse_loss(pred, outputs)
        pen = 0
        for i in range(len(hidden)-1):
            pen += torch.sum((hidden[i+1] - hidden[i]).abs())
        return mse + penalty_factor*pen

    def _train_model(self, inputs, outputs, window_size):
        """
        Trains an autoencoder model from inputs and outputs
        """
        loss_function = self._custom_loss
        penalty_factor = self._get_penalty_factor(inputs.shape[0])

        model =  _Autoencoder(window_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        losses = []
        for epoch in range(self.epochs):
            curr_loss = 0
            model.zero_grad()
            loss = 0
            pred, hidden = model(inputs)
            loss = loss_function(pred, outputs, hidden, penalty_factor)
            loss.backward()
            optimizer.step()
            losses.append(loss)

        with torch.no_grad():
            predictions, reduced_windows = model(inputs)
        return predictions.detach().numpy().reshape(-1), \
            reduced_windows.detach().numpy().reshape(-1, window_size//16)

    def _find_breakpoints(self, reduced_windows, window_size):
        """
        Finds the breakpoints from the hidden layer using Agglomerative clustering
        """
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import pdist
        from sklearn.metrics import silhouette_score

        labels = np.zeros(reduced_windows.shape[0])
        best_silouhette, best_nb = self.min_silhouette_score, 1
        for i in range(2, 6):
            km = KMeans(n_clusters=i)
            tmp_labels = km.fit_predict(reduced_windows)
            if min(pdist(km.cluster_centers_)) < self.distance_threshold:
                break
            silhouette = silhouette_score(reduced_windows, tmp_labels)
            if silhouette > best_silouhette:
                labels = tmp_labels
                best_silouhette, best_nb = silhouette, i

        change_points = []
        labels_cp = [labels[0]]
        cnt = 1
        for clus_prev, clus in zip(labels[:-1], labels[1:]):
            if clus != clus_prev:
                change_points.append(cnt*window_size)
                labels_cp.append(clus)
            cnt += 1
        if len(change_points) != 0:
            return change_points, labels_cp
        return (None, None)

    def run(self, ts, period):
        """Running autoencoder to detect baseline change
        
        Arguments:
            ts {TimeSeries} -- TimeSeries on which we want to detect the change
            period {int} -- Seasonality of the timeseries
        
        Returns:
            Tuple -- (Change points indexes, clusters)
        """
        window_size = self._get_window_size(ts, period)
        ts_scaled = self._scale_ts(ts)
        inputs, outputs = self._ts_to_input(ts_scaled, window_size)
        _, reduced = self._train_model(inputs, outputs, window_size)
        return self._find_breakpoints(reduced, window_size)


def run_atr_wrapper(window_size, penalty_coefficient=PENALTY_COEFFICIENT, weight_decay=WEIGHT_DECAY,
            distance_threshold=DISTANCE_THRESHOLD, min_silhouette_score=MIN_SILHOUETTE_SCORE,
            learning_rate=LEARNING_RATE):
    def run_atr(ts):
        atr = ATR(penalty_coefficient=penalty_coefficient, weight_decay=weight_decay,
                        distance_threshold=distance_threshold, min_silhouette_score=min_silhouette_score,
                        learning_rate=learning_rate)
        return atr.run(ts, window_size)
    return run_atr
