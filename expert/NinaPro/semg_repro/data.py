# built in libraries
import random
import multiprocessing

# third party
import numpy as np
from scipy import signal
from scipy.io import loadmat

# local
from label_dict import label_dict
from bc_dict import bc_dict

# build window rolling scheme
def roll_labels(x, y):
    labs_rolled = []
    for i in range(len(y)):
        l = y[i]
        n = x[i].shape[0]
        labs_rolled.append(np.repeat(l, n))
    return np.hstack(labs_rolled)


def window_roll(a, stepsize=5, width=52):
    n = a.shape[0]
    emg = np.dstack([a[i : 1 + n + i - width : stepsize] for i in range(0, width)])
    return emg


# build augmentation scheme
def add_noise_snr(signal, snr=25):
    # convert signal to db
    sgn_db = np.log10((signal ** 2).mean(axis=0)) * 10
    # noise in db
    noise_avg_db = sgn_db - snr
    # convert noise_db
    noise_variance = 10 ** (noise_avg_db / 10)
    # make some white noise using this as std
    noise = np.random.normal(0, np.sqrt(noise_variance), signal.shape)
    return signal + noise


# noise factors to sample from, outside of the function because this will be
# called millions of times
rlist = sum([[(x / 2) % 30] * ((x // 2) % 30) for x in range(120)], [])


def add_noise_random(signal):
    num = random.choice(rlist)
    return add_noise_snr(signal, num)


# moving average
def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode="valid")


def ma(window, n):
    return np.vstack(
        [moving_average(window[:, i], n) for i in range(window.shape[-1])]
    ).T


def ma_batch(batch, n):
    return np.dstack([ma(batch[i, :, :], n) for i in range(batch.shape[0])])


# butter filter preprocess
def _butter_highpass(cutoff, fs, order=3):
    # nyquist frequency!!
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_highpass_filter(data, cutoff=2, fs=200, order=3):
    b, a = _butter_highpass(cutoff=cutoff, fs=fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


# dataset loading class:
# first some helpers:
def first0(x):
    return np.unique(x)[0]


def first_appearance(arr):
    # gets the first class in the case of overlapping due to our windowing
    inn = [arr[i] for i in range(arr.shape[0])]
    with multiprocessing.Pool(None) as p:
        res = p.map(first0, inn)
    return np.asarray(res)


class dataset(object):
    def __init__(
        self,
        path,
        butter=True,
        rectify=True,
        ma=15,
        step=5,
        window=52,
        exercises=["a", "b", "c"],
        features=None,
    ):
        self.path = path
        self.butter = butter
        self.rectify = rectify
        self.ma = ma
        self.step = step
        self.window = window
        self.exercises = exercises
        self.features = features

        # load the data
        self.read_data()
        self.process_data()

    def _load_file(self, path, ex, features=None):
        """
        loads a file given a path, and relabels it according to the exercise dict
        provided in label_dict. Each set of trials has labels starting at 0,
        which needs to be corrected
        """
        res = loadmat(path)
        data = []
        # imu data
        imu = res["acc"].copy()
        # repetition labeled by a machine (more accurate labels, this is what we
        # will use to split the data by)
        rep = res["rerepetition"].copy()
        # emg data
        emg = res["emg"].copy()
        # machine labeled exercises
        lab = res["restimulus"].copy()
        # relabel 0:52
        if 'a' not in self.exercises:
            lab = np.array([[bc_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])
        else:
            lab = np.array([[label_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])

        del res
        # make it possible to engineer features

        data.append(emg)
        if features:
            for ft in features:
                print("adding features")
                sameDim = data[0].shape[0] == np.shape(res[ft])[0]
                newData = []
                if not sameDim and np.shape(res[ft])[1] == 1:
                    newData = np.full((np.shape(data[0])[0], 1), res[ft][0, 0])
                else:
                    newData = res[ft]
                data.append(newData)

        return np.concatenate(data, axis=1), lab, rep, imu

    def _load_by_trial(self, trial=1, features=None):
        data = []
        labs = []
        reps = []
        imu = []
        for i in range(1, 11):
            path = f"{self.path}/s{i}/S{i}_E{trial}_A1.mat"
            emg, l, r, ii = self._load_file(path, ex=trial, features=features)
            data.append(emg)
            labs.append(l)
            reps.append(r)
            imu.append(ii)
        return data, labs, reps, imu

    def read_data(self):
        ex_dict = dict(zip(["a", "b", "c"], range(1, 4)))
        self.emg = []
        self.labels = []
        self.repetition = []
        self.imu = []

        for e in self.exercises:
            # In the papers the exercises are lettered not numbered, but to load
            # the data properly we need them to be numbered. an exercise
            # represents a group of either hand motions, funcitonal motions, or
            # wrist motions
            exercise = ex_dict[e]
            emg, lab, rep, imu = self._load_by_trial(trial=exercise, features=self.features)
            self.emg += emg
            self.labels += lab
            self.repetition += rep
            self.imu += imu
        print(sum([x.shape[0] for x in self.emg]))

    def process_data(self):
        if self.rectify:
            self.emg = [np.abs(x) for x in self.emg]

        if self.butter:
            self.emg = [butter_highpass_filter(x) for x in self.emg]

        self.flat = [self.emg, self.labels, self.repetition, self.imu]
        self.emg = [window_roll(x, self.step, self.window) for x in self.emg]
        self.imu = [window_roll(x, self.step, self.window) for x in self.imu]
        self.labels = [window_roll(x, self.step, self.window) for x in self.labels]
        self.repetition = [window_roll(x, self.step, self.window) for x in self.repetition]

        # reshape the data to have the axes in the proper order
        self.emg = np.moveaxis(np.concatenate(self.emg, axis=0), 2, 1)
        self.imu = np.moveaxis(np.concatenate(self.imu, axis=0), 2, 1)
        self.labels = np.moveaxis(np.concatenate(self.labels, axis=0), 2, 1)[..., -1]
        self.repetition = np.moveaxis(np.concatenate(self.repetition, axis=0), 2, 1)[..., -1]

        # we split by repetition, and we do not want any data leaks. So, we
        # simply drop any window that has more than one repetition in it
        no_leaks = np.array(
            [
                i
                for i in range(self.repetition.shape[0])
                if np.unique(self.repetition[i]).shape[0] == 1
            ]
        )



        self.emg = self.emg[no_leaks, :, :]
        self.imu = self.imu[no_leaks, :, :]
        self.labels = self.labels[no_leaks, :]
        self.repetition = self.repetition[no_leaks, :]

        # next we want to make sure there arent multiple labels. We do this
        # using the first class that appears in a window. Intuitively, this
        # makes sense, as when someone is grabbing something then finishes
        # halfway through, they still completed the act of grabbing something


        self.labels = first_appearance(self.labels)
        self.repetition = first_appearance(self.repetition)
        self.emg = self.emg.astype(np.float16)
        self.imu = self.imu.astype(np.float16)



class nina4_dataset(dataset):
    def __init__(
        self,
        path,
        butter=True,
        rectify=True,
        ma=15,
        step=5,
        window=52,
        exercises=["a", "b", "c"],
        features=None,
        n_subjects=10
    ):
        self.path = path
        self.n_subjects = n_subjects
        self.butter = butter
        self.rectify = rectify
        self.ma = ma
        self.step = step
        self.window = window
        self.exercises = exercises
        self.features = features

        # load the data
        print("reading")
        self.read_data()
        print("processing")
        self.process_data()

    def _load_file(self, path, ex, features=None):
        """
        loads a file given a path, and relabels it according to the exercise dict
        provided in label_dict. Each set of trials has labels starting at 0,
        which needs to be corrected
        """
        res = loadmat(path)
        data = []
        # repetition labeled by a machine (more accurate labels, this is what we
        # will use to split the data by)
        rep = res["rerepetition"].copy()
        # emg data
        emg = res["emg"].copy()
        # machine labeled exercises
        lab = res["restimulus"].copy()
        # relabel 0:52
        lab = np.array([[label_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])

        del res
        # make it possible to engineer features

        data.append(emg)
        if features:
            for ft in features:
                print("adding features")
                sameDim = data[0].shape[0] == np.shape(res[ft])[0]
                newData = []
                if not sameDim and np.shape(res[ft])[1] == 1:
                    newData = np.full((np.shape(data[0])[0], 1), res[ft][0, 0])
                else:
                    newData = res[ft]
                data.append(newData)

        return np.concatenate(data, axis=1), lab, rep

    def _load_by_trial(self, trial=1, features=None):
        data = []
        labs = []
        reps = []
        for i in range(1, self.n_subjects+1):
            path = f"{self.path}/s{i}/S{i}_E{trial}_A1.mat"
            emg, l, r = self._load_file(path, ex=trial, features=features)
            data.append(emg)
            labs.append(l)
            reps.append(r)
        return data, labs, reps

    def read_data(self):
        ex_dict = dict(zip(["a", "b", "c"], range(1, 4)))
        self.emg = []
        self.labels = []
        self.repetition = []

        for e in self.exercises:
            # In the papers the exercises are lettered not numbered, but to load
            # the data properly we need them to be numbered. an exercise
            # represents a group of either hand motions, funcitonal motions, or
            # wrist motions
            exercise = ex_dict[e]
            emg, lab, rep = self._load_by_trial(trial=exercise, features=self.features)
            self.emg += emg
            self.labels += lab
            self.repetition += rep

    def process_data(self):
        if self.rectify:
            self.emg = [np.abs(x) for x in self.emg]

        if self.butter:
            self.emg = [butter_highpass_filter(x) for x in self.emg]

        print("rolling")
        self.emg = [window_roll(x, self.step, self.window) for x in self.emg]
        self.labels = [window_roll(x, self.step, self.window) for x in self.labels]
        self.repetition = [window_roll(x, self.step, self.window) for x in self.repetition]

        # reshape the data to have the axes in the proper order
        self.emg = np.moveaxis(np.concatenate(self.emg, axis=0), 2, 1)
        self.labels = np.moveaxis(np.concatenate(self.labels, axis=0), 2, 1)[..., -1]
        self.repetition = np.moveaxis(np.concatenate(self.repetition, axis=0), 2, 1)[..., -1]

        # we split by repetition, and we do not want any data leaks. So, we
        # simply drop any window that has more than one repetition in it
        no_leaks = np.array(
            [
                i
                for i in range(self.repetition.shape[0])
                if np.unique(self.repetition[i]).shape[0] == 1
            ]
        )

        self.emg = self.emg[no_leaks, :, :]
        self.labels = self.labels[no_leaks, :]
        self.repetition = self.repetition[no_leaks, :]

        # next we want to make sure there arent multiple labels. We do this
        # using the first class that appears in a window. Intuitively, this
        # makes sense, as when someone is grabbing something then finishes
        # halfway through, they still completed the act of grabbing something

        print("cleaning")
        self.labels = first_appearance(self.labels)
        self.repetition = first_appearance(self.repetition)
        self.emg = self.emg.astype(np.float16)



class nina1_dataset(dataset):
    def __init__(
        self,
        path,
        butter=True,
        rectify=True,
        ma=15,
        step=5,
        window=52,
        exercises=["a", "b", "c"],
        features=None,
        n_subjects=27
    ):
        self.path = path
        self.n_subjects = n_subjects
        self.butter = butter
        self.rectify = rectify
        self.ma = ma
        self.step = step
        self.window = window
        self.exercises = exercises
        self.features = features

        # load the data
        print("reading")
        self.read_data()
        print("processing")
        self.process_data()

    def _load_file(self, path, ex, features=None):
        """
        loads a file given a path, and relabels it according to the exercise dict
        provided in label_dict. Each set of trials has labels starting at 0,
        which needs to be corrected
        """
        res = loadmat(path)
        data = []
        # repetition labeled by a machine (more accurate labels, this is what we
        # will use to split the data by)
        rep = res["rerepetition"].copy()
        # emg data
        emg = res["emg"].copy()
        # machine labeled exercises
        lab = res["restimulus"].copy()
        # relabel 0:52
        lab = np.array([[label_dict[ex][lab[i][0]]] for i in range(lab.shape[0])])

        del res
        # make it possible to engineer features

        data.append(emg)
        if features:
            for ft in features:
                print("adding features")
                sameDim = data[0].shape[0] == np.shape(res[ft])[0]
                newData = []
                if not sameDim and np.shape(res[ft])[1] == 1:
                    newData = np.full((np.shape(data[0])[0], 1), res[ft][0, 0])
                else:
                    newData = res[ft]
                data.append(newData)

        return np.concatenate(data, axis=1), lab, rep

    def _load_by_trial(self, trial=1, features=None):
        data = []
        labs = []
        reps = []
        for i in range(1, self.n_subjects+1):
            path = f"{self.path}/s{i}/S{i}_A1_E{trial}.mat"
            emg, l, r = self._load_file(path, ex=trial, features=features)
            data.append(emg)
            labs.append(l)
            reps.append(r)
        return data, labs, reps

    def read_data(self):
        ex_dict = dict(zip(["a", "b", "c"], range(1, 4)))
        self.emg = []
        self.labels = []
        self.repetition = []

        for e in self.exercises:
            # In the papers the exercises are lettered not numbered, but to load
            # the data properly we need them to be numbered. an exercise
            # represents a group of either hand motions, funcitonal motions, or
            # wrist motions
            exercise = ex_dict[e]
            emg, lab, rep = self._load_by_trial(trial=exercise, features=self.features)
            self.emg += emg
            self.labels += lab
            self.repetition += rep

    def process_data(self):
        if self.rectify:
            self.emg = [np.abs(x) for x in self.emg]

        if self.butter:
            self.emg = [butter_highpass_filter(x) for x in self.emg]

        print("rolling")
        self.emg = [window_roll(x, self.step, self.window) for x in self.emg]
        self.labels = [window_roll(x, self.step, self.window) for x in self.labels]
        self.repetition = [window_roll(x, self.step, self.window) for x in self.repetition]

        # reshape the data to have the axes in the proper order
        self.emg = np.moveaxis(np.concatenate(self.emg, axis=0), 2, 1)
        self.labels = np.moveaxis(np.concatenate(self.labels, axis=0), 2, 1)[..., -1]
        self.repetition = np.moveaxis(np.concatenate(self.repetition, axis=0), 2, 1)[..., -1]

        # we split by repetition, and we do not want any data leaks. So, we
        # simply drop any window that has more than one repetition in it
        no_leaks = np.array(
            [
                i
                for i in range(self.repetition.shape[0])
                if np.unique(self.repetition[i]).shape[0] == 1
            ]
        )


        self.emg = self.emg[no_leaks, :, :]
        self.labels = self.labels[no_leaks, :]
        self.repetition = self.repetition[no_leaks, :]

        # next we want to make sure there arent multiple labels. We do this
        # using the first class that appears in a window. Intuitively, this
        # makes sense, as when someone is grabbing something then finishes
        # halfway through, they still completed the act of grabbing something

        print("cleaning")
        self.labels = first_appearance(self.labels)
        self.repetition = first_appearance(self.repetition)
        self.emg = self.emg.astype(np.float16)

        self.emg = self.emg[np.where(self.labels != 0)[0]]
        self.repetition = self.repetition[np.where(self.labels != 0)[0]]
        self.labels = self.labels[np.where(self.labels !=0)[0]]
        self.labels -= 1

