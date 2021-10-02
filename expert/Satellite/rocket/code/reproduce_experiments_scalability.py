# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)
import os
import argparse
import numpy as np
import pandas as pd
import time
import torch, torch.nn as nn, torch.optim as optim

from rocket_functions import apply_kernels, generate_kernels

# == notes =====================================================================

# Reproduce the scalability experiments.
#
# Arguments:
# -tr --training_path : training dataset (npy)
# -te --test_path     : test dataset (npy)
# -o  --output_path   : path for results
# -k  --num_kernels   : number of kernels

# == parse arguments ===========================================================

parser = argparse.ArgumentParser()

parser.add_argument("-path", "--data_path", required = True)
parser.add_argument("-o", "--output_path", required = True)
parser.add_argument("-k", "--num_kernels", type = int)
parser.add_argument("-seed", "--seed", type=int)

arguments = parser.parse_args()

# == training function =========================================================

def train(X,
          Y,
          X_validation,
          Y_validation,
          kernels,
          num_features,
          num_classes,
          minibatch_size = 256,
          max_epochs = 100,
          patience = 2,           # x10 minibatches; reset if loss improves
          tranche_size = 2 ** 11,
          cache_size = 2 ** 14):  # as much as possible

    # -- init ------------------------------------------------------------------

    def init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.weight.data, 0)
            nn.init.constant_(layer.bias.data, 0)

    # -- model -----------------------------------------------------------------

    model = nn.Sequential(nn.Linear(num_features, num_classes)) # logistic / softmax regression
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, min_lr = 1e-8)
    model.apply(init)

    # -- run -------------------------------------------------------------------

    minibatch_count = 0
    best_validation_loss = np.inf
    stall_count = 0
    stop = False

    num_examples = len(X)
    num_tranches = np.int(np.ceil(num_examples / tranche_size))

    cache = np.zeros((min(cache_size, num_examples), num_features))
    cache_count = 0

    for epoch in range(max_epochs):

        if epoch > 0 and stop:
            break

        for tranche_index in range(num_tranches):

            if epoch > 0 and stop:
                break

            a = tranche_size * tranche_index
            b = a + tranche_size

            Y_tranche = Y[a:b]

            # if cached, use cached transform; else transform and cache the result
            if b <= cache_count:

                X_tranche_transform = cache[a:b]

            else:

                X_tranche = X[a:b]
                X_tranche = (X_tranche - X_tranche.mean(axis = 1, keepdims = True)) / X_tranche.std(axis = 1, keepdims = True) # normalise time series
                X_tranche_transform = apply_kernels(X_tranche, kernels)

                if epoch == 0 and tranche_index == 0:

                    # per-feature mean and standard deviation (estimated on first tranche)
                    f_mean = X_tranche_transform.mean(0)
                    f_std = X_tranche_transform.std(0) + 1e-8

                    # normalise and transform validation data
                    X_validation = (X_validation - X_validation.mean(axis = 1, keepdims = True)) / X_validation.std(axis = 1, keepdims = True) # normalise time series
                    X_validation_transform = apply_kernels(X_validation, kernels)
                    X_validation_transform = (X_validation_transform - f_mean) / f_std # normalise transformed features
                    X_validation_transform = torch.FloatTensor(X_validation_transform)
                    Y_validation = torch.LongTensor(Y_validation)

                X_tranche_transform = (X_tranche_transform - f_mean) / f_std # normalise transformed features

                if b <= cache_size:

                    cache[a:b] = X_tranche_transform
                    cache_count = b

            X_tranche_transform = torch.FloatTensor(X_tranche_transform)
            Y_tranche = torch.LongTensor(Y_tranche)

            minibatches = torch.randperm(len(X_tranche_transform)).split(minibatch_size)

            for minibatch_index, minibatch in enumerate(minibatches):

                if epoch > 0 and stop:
                    break

                # abandon undersized minibatches
                if minibatch_index > 0 and len(minibatch) < minibatch_size:
                    break

                # -- (optional) minimal lr search ------------------------------

                # default lr for Adam may cause training loss to diverge for a
                # large number of kernels; lr minimising training loss on first
                # update should ensure training loss converges

                if epoch == 0 and tranche_index == 0 and minibatch_index == 0:

                    candidate_lr = 10 ** np.linspace(-1, -6, 6)

                    best_lr = None
                    best_training_loss = np.inf

                    for lr in candidate_lr:

                        lr_model = nn.Sequential(nn.Linear(num_features, num_classes))
                        lr_optimizer = optim.Adam(lr_model.parameters())
                        lr_model.apply(init)

                        for param_group in lr_optimizer.param_groups:
                            param_group["lr"] = lr

                        # perform a single update
                        lr_optimizer.zero_grad()
                        Y_tranche_predictions = lr_model(X_tranche_transform[minibatch])
                        training_loss = loss_function(Y_tranche_predictions, Y_tranche[minibatch])
                        training_loss.backward()
                        lr_optimizer.step()

                        Y_tranche_predictions = lr_model(X_tranche_transform)
                        training_loss = loss_function(Y_tranche_predictions, Y_tranche).item()

                        if training_loss < best_training_loss:
                            best_training_loss = training_loss
                            best_lr = lr

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = best_lr

                # -- training --------------------------------------------------

                optimizer.zero_grad()
                Y_tranche_predictions = model(X_tranche_transform[minibatch])
                training_loss = loss_function(Y_tranche_predictions, Y_tranche[minibatch])
                training_loss.backward()
                optimizer.step()

                minibatch_count += 1

                if minibatch_count % 10 == 0:

                    Y_validation_predictions = model(X_validation_transform)
                    validation_loss = loss_function(Y_validation_predictions, Y_validation)

                    scheduler.step(validation_loss)

                    if validation_loss.item() >= best_validation_loss:
                        stall_count += 1
                        if stall_count >= patience:
                            stop = True
                    else:
                        best_validation_loss = validation_loss.item()
                        if not stop:
                            stall_count = 0

    return model, f_mean, f_std

# == run =======================================================================

# -- run through dataset sizes -------------------------------------------------
np.random.seed(arguments.seed)
torch.manual_seed(arguments.seed)

all_num_training_examples = [900000]

results = pd.DataFrame(index = all_num_training_examples,
                       columns = ["accuracy", "time_training_seconds"],
                       data = 0)
results.index.name = "num_training_examples"

print(f" {arguments.num_kernels:,} Kernels ".center(80, "="))

for num_training_examples in all_num_training_examples:

    if num_training_examples == all_num_training_examples[0]:
        print("Number of training examples:" + f"{num_training_examples:,}".rjust(75 - 28 - 5, " ") + ".....", end = "", flush = True)
    else:
        print(f"{num_training_examples:,}".rjust(75 - 5, " ") + ".....", end = "", flush = True)

    # -- read training and validation data -------------------------------------

    # if training data does not fit in memory, it is possible to load the
    # training data inside the train(...) function, using the *chunksize*
    # argument for pandas.read_csv(...) (and roughly substituting chunks for
    # tranches); similarly, if the cache does not fit in memory, consider
    # caching the transformed features on disk
    
    path = arguments.data_path
    train_file = os.path.join(path, 'satellite_train.npy')
    test_file = os.path.join(path, 'satellite_test.npy')

    X_training, Y_training = np.load(train_file, allow_pickle=True)[()]['data'], np.load(train_file,allow_pickle=True)[()]['label']
    X_validation, Y_validation = np.load(test_file, allow_pickle=True)[()]['data'], np.load(test_file, allow_pickle=True)[()]['label']
    Y_training = Y_training - 1
    Y_validation = Y_validation - 1
    print(np.unique(Y_training))

    # -- generate kernels ------------------------------------------------------

    kernels = generate_kernels(X_training.shape[1], arguments.num_kernels)

    # -- train -----------------------------------------------------------------

    time_a = time.perf_counter()
    model, f_mean, f_std = train(X_training,
                                 Y_training,
                                 X_validation,
                                 Y_validation,
                                 kernels,
                                 arguments.num_kernels * 2,
                                 num_classes = 24)
    time_b = time.perf_counter()

    results.loc[num_training_examples, "time_training_seconds"] = time_b - time_a

    # -- test ------------------------------------------------------------------

    # read test data (here, we test on a subset of the full test data)
    X_test, Y_test = X_validation, Y_validation

    # normalise and transform test data
    X_test = (X_test - X_test.mean(axis = 1, keepdims = True)) / X_test.std(axis = 1, keepdims = True) # normalise time series
    X_test_transform = apply_kernels(X_test, kernels)
    X_test_transform = (X_test_transform - f_mean) / f_std # normalise transformed features

    # predict
    model.eval()
    Y_test_predictions = model(torch.FloatTensor(X_test_transform))

    results.loc[num_training_examples, "accuracy"] = (Y_test_predictions.max(1)[1].numpy() == Y_test).mean()

    print("Done.")

print(f" FINISHED ".center(80, "="))

results.to_csv(f"{arguments.output_path}/results_scalability_k={arguments.num_kernels}.csv")
