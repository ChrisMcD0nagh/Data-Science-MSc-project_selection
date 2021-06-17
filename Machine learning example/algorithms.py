import numpy as np
from scipy.spatial import distance
import random
import concurrent.futures
import time


def RMSE(pred, y_test):
    """Calculate the root mean squared error of predictions and test data."""
    pred = pred.reshape(len(pred))
    y_test = y_test.reshape(len(y_test))
    return (sum((pred - y_test)**2) / len(y_test))**0.5


# Linear regression...

def linear_regression(
        x_train,
        y_train,
        x_test,
        y_test,
        lambda_,
        add_constant=True):
    """ Calculate linear regressoin coefficients and evaluate RMSE."""
    if add_constant:
        x_train_c = np.zeros((x_train.shape[0], x_train.shape[1] + 1))
        x_test_c = np.zeros((x_test.shape[0], x_test.shape[1] + 1))
        x_train_c[:, :-1] = x_train
        x_test_c[:, :-1] = x_test
        x_train_c[:, -1] = 1
        x_test_c[:, -1] = 1
        x_train = x_train_c
        x_test = x_test_c

    coefs = np.linalg.solve(
        (x_train.T @ x_train +
         lambda_ *
         np.eye(
             x_train.shape[1])),
        x_train.T @ y_train)
    train_pred = np.dot(x_train, coefs)
    test_pred = np.dot(x_test, coefs)
    train_RMSE = RMSE(train_pred, y_train)
    test_RMSE = RMSE(test_pred, y_test)
    results = {'Train RMSE': train_RMSE,
               'Test RMSE': test_RMSE,
               'coefs': coefs,
               'train_preds': train_pred,
               'test_preds': test_pred}
    return results


# Regression forest...

def calc_variance(y):
    """ Find the variance of y."""
    if len(y) == 0:
        return np.inf
    var = np.var(y)
    return var


def find_split(x, y, alpha):
    """Given a dataset and its target values, this finds the optimal combination
    of feature and split point that gives the maximum information gain."""

    # Need the starting entropy so we can measure improvement...
    start_variance = calc_variance(y)

    # Best thus far, initialised to a dud that will be replaced immediately...
    best = {'Variance reduction': -np.inf}

    # Loop every possible split over a random selection of root f dimensions...
    features_to_select = int(x.shape[1]**0.5)
    features = random.sample(range(0, x.shape[1]), features_to_select)

    for f in features:
        feature_min_val = min(np.unique(x[:, f]))
        feature_max_val = max(np.unique(x[:, f]))

        # Round up to avoid 0 splits
        no_splits_to_check = int(
            len(np.unique(x[:, f])) / alpha) + (len(np.unique(x[:, f]) % alpha > 0))
        splits = np.linspace(
            feature_min_val,
            feature_max_val,
            no_splits_to_check)
        for split in splits:
            left_indices_bool = x[:, f] <= split
            left_indices = [i for i, k in enumerate(left_indices_bool) if k]
            right_indices_bool = x[:, f] > split
            right_indices = [i for i, k in enumerate(right_indices_bool) if k]
            n_left = len(left_indices)
            n_right = len(right_indices)
            left_variance = calc_variance(y[left_indices])
            right_variance = calc_variance(y[right_indices])
            left_weight = n_left / len(y)
            right_weight = n_right / len(y)
            var_reduction = start_variance - \
                (left_weight * left_variance) - (right_weight * right_variance)

            if var_reduction > best['Variance reduction']:
                best = {'feature': f,
                        'split': split,
                        'Variance reduction': var_reduction,
                        'left_indices': left_indices,
                        'right_indices': right_indices}
    return best


def build_tree(x, y, alpha, gamma, max_depth=np.inf):
    """ Build a tree with the given data."""
    # Check that all rows of x values are not identical...
    features = x.shape[1]
    all_vals_identical = 1
    for i in range(features):
        if len(np.unique(x[:, i])) == 1:
            all_vals_identical *= 0
    # Check if either of the stopping conditions have been reached. If so
    # generate a leaf node...
    if max_depth == 1 or calc_variance(
            y) < 1e-1 or len(y) <= 100 or all_vals_identical == 0:
        # Generate a leaf node...
        return {'leaf': True, 'prediction': y.mean()}
    else:
        move = find_split(x, y, alpha)
        left = build_tree(x[move['left_indices'], :],
                          y[move['left_indices']], alpha, gamma, max_depth - 1)
        right = build_tree(x[move['right_indices'], :],
                           y[move['right_indices']], alpha, gamma, max_depth - 1)

        return {'leaf': False,
                'feature': move['feature'],
                'split': move['split'],
                'Variance reduction': move['Variance reduction'],
                'left': left,
                'right': right}


def predict(tree, samples):
    """Predicts class for every entry of a data matrix."""
    ret = np.empty(samples.shape[0], dtype=float)
    ret.fill(-1)
    indices = np.arange(samples.shape[0])

    def tranverse(node, indices):
        nonlocal samples
        nonlocal ret

        if node['leaf']:
            ret[indices] = node['prediction']

        else:
            going_left = samples[indices, node['feature']] <= node['split']
            left_indices = indices[going_left]
            right_indices = indices[np.logical_not(going_left)]

            if left_indices.shape[0] > 0:
                tranverse(node['left'], left_indices)

            if right_indices.shape[0] > 0:
                tranverse(node['right'], right_indices)

    tranverse(tree, indices)
    return ret


def evaluate(x_train, y_train, x_test, y_test, hps, random_states, k=1000):
    """Evaluate trees on test, train and out of bag data."""
    max_depth = int(hps[0])
    alpha = hps[1]
    gamma = hps[2]

    predictions_train = np.zeros(x_train.shape[0])
    predictions_OOB = np.zeros(x_train.shape[0])
    count_train = np.zeros(x_train.shape[0])
    count_OOB = np.zeros(x_train.shape[0])
    predictions_test = np.zeros(y_test.shape[0])

    for rs in random_states:
        random.seed(rs)
        train_indicies = random.choices(np.arange(x_train.shape[0]), k=k)
        all_indicies = np.arange(x_train.shape[0])
        OOB_indicies = list(set(all_indicies) - set(train_indicies))

        x_train_trees = x_train[train_indicies]
        y_train_trees = y_train[train_indicies]
        x_OOB = x_train[OOB_indicies]

        # Build the tree with the training data...
#         print(f'Buidling tree with random state {rs}')
        tree = build_tree(x_train_trees, y_train_trees, alpha, gamma, max_depth)
#         print('Built tree')
        # Assess train accuracy...
        tree_predict_train = predict(tree, x_train_trees)
        predictions_train[train_indicies] += tree_predict_train
        count_train[train_indicies] += 1

        # Assess OOB accuracy...
        tree_predict_OOB = predict(tree, x_OOB)
        predictions_OOB[OOB_indicies] += tree_predict_OOB
        count_OOB[OOB_indicies] += 1

        # Assess test accuracy...
        tree_predict_test = predict(tree, x_test)
        predictions_test += tree_predict_test

    predictions_train_avg = np.divide(
        predictions_train,
        count_train,
        out=np.zeros_like(predictions_train),
        where=count_train != 0)
    predictions_OOB_avg = np.divide(
        predictions_OOB,
        count_OOB,
        out=np.zeros_like(predictions_OOB),
        where=count_OOB != 0)
    predictions_test_avg = predictions_test / len(random_states)
    RMSE_train = RMSE(
        predictions_train_avg.reshape(
            len(predictions_train_avg), 1), y_train.reshape(
            len(y_train), 1))
    RMSE_OOB = RMSE(
        predictions_OOB_avg.reshape(
            len(predictions_OOB_avg), 1), y_train.reshape(
            len(y_train), 1))
    RMSE_test = RMSE(
        predictions_test_avg.reshape(
            len(predictions_test_avg), 1), y_test.reshape(
            len(y_test), 1))
    return RMSE_train, RMSE_OOB, RMSE_test, predictions_test_avg


def regression_forest(x_train,
                      y_train,
                      x_test,
                      y_test,
                      hps,
                      rtn_RMSE=True,
                      kwargs={'k': 1000, 'no_trees': 10, 'run_concurrent': False}):
    """ Build and evaluate regression forests given then training and test data."""

    # Assign key word arguments to variables...
    globals().update(kwargs)
    random.seed(42)
    random_states = random.sample(range(0, 9999), no_trees)
    if run_concurrent:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result1 = executor.submit(evaluate,
                                      x_train,
                                      y_train,
                                      x_test,
                                      y_test,
                                      hps,
                                      random_states[::2],
                                      k=k)
            result2 = executor.submit(evaluate,
                                      x_train,
                                      y_train,
                                      x_test,
                                      y_test,
                                      hps,
                                      random_states[1::2],
                                      k=k)
            RMSE_train = (result1.result()[0] + result2.result()[0]) / 2
            RMSE_OOB = (result1.result()[1] + result2.result()[1]) / 2
            RMSE_test = (result1.result()[2] + result2.result()[2]) / 2
            test_preds = (result1.result()[3] + result2.result()[3]) / 2
    else:
        result = evaluate(
            x_train,
            y_train,
            x_test,
            y_test,
            hps,
            random_states,
            k=k)
        RMSE_train = result[0]
        RMSE_OOB = result[1]
        RMSE_test = result[2]
        test_preds = result[3]

    results = {'Train RMSE': RMSE_train,
               'OOB RMSE': RMSE_OOB,
               'Test RMSE': RMSE_test,
               'test_pred': test_preds}
    return results


# Gaussain process...

def gaussian_kernel(x, y, l2=0.1, gk_var=1):
    """Gaussian kernal (squared exponential) with lengthscale and amplitude hyperparameters"""
    sqdist = np.sum(x**2, 1).reshape(-1, 1) + \
        np.sum(y**2, 1) - 2 * np.dot(x, y.T)
    return gk_var * np.exp(-.5 * (1 / l2) * sqdist)


def fit_gaussian(
        x_train,
        y_train,
        x_predict,
        y_test,
        hps,
        rtn_RMSE=False,
        kwargs=None):
    """Fit a gaussian process to the data."""
    assert x_train.shape[0] <= 15000, "Training data too large (intractable)."
    l2 = hps[0]
    gk_var = hps[1]
    noise_var = hps[2]
    n_train = len(x_train)
    K = gaussian_kernel(x_train, x_train, l2, gk_var)
    start = time.time()
    L = np.linalg.cholesky(K + noise_var * np.eye(n_train))
    end = time.time() - start
    # print(f'mat inverted in {end:.02f} secs')
    K_ = gaussian_kernel(x_predict, x_predict, l2, gk_var)
    Lk = np.linalg.solve(L, gaussian_kernel(x_train, x_predict, l2, gk_var))
    mu = np.dot(
        Lk.T, np.linalg.solve(
            L, (y_train - np.mean(y_train)))) + np.mean(y_train)
    sigma = np.sqrt(np.diag(K_) - np.sum(Lk**2, axis=0))

    if rtn_RMSE:
        results = {'Test RMSE': RMSE(mu, y_test),
                   'test_preds': mu}
        return results
    else:
        return mu, sigma


# Weighted K-means...

def k_means(
        x_train,
        y_train,
        x_test,
        y_test,
        k,
        steps=10,
        dist_weight=True):
    """ Fit and evaluate k-means clusters on data"""
    assert k <= x_train.shape[0], "K must be less than the number of traning data."

    # Force k to be an integer...
    k = int(k)
   

    # Choose k random clusters to begin...
    np.random.seed(42)
    index = np.random.choice(x_train.shape[0], k, replace=False)
    centroids = x_train[index, :]

    # Find indicies of data closest to each centroid...
    P = np.argmin(distance.cdist(x_train, centroids, 'euclidean'), axis=1)

    for _ in range(steps):
        # Create new centroids at the means of each centroid...
        centroids = np.vstack([np.mean(x_train[P == i, :], axis=0)
                               for i in range(k) if len(x_train[P == i, :]) != 0])

        # Find indicies of data closests to each (new) centroid...
        temp_index = np.argmin(
            distance.cdist(
                x_train,
                centroids,
                'euclidean'),
            axis=1)

        # If no data changes centroid, stop...
        if np.array_equal(P, temp_index):
            break

        # Repeat process with new indicies...
        P = temp_index

    # Find the predictions (weighted average) from each cluster...
    P = np.argmin(distance.cdist(x_train, centroids, 'euclidean'), axis=1)
    distances = np.min(distance.cdist(x_train, centroids, 'euclidean'), axis=1)
    cluster_preds = np.zeros(k)
    total_distances = np.stack(
        [np.sum(1 / (distances[P == i])) for i in range(k)])
    if dist_weight:
        # Handle 0 distances...
        e = 1e-6
    
        # Find predictions weighted by inverse of distances
        for j in range(k):
            for i in range(len(distances[P == j])):
                cluster_preds[j] += (1 / (distances[P == j][i] + e)) * \
                    y_train[P == j][i] / (total_distances[j])
            
    else:       
        cluster_preds = np.stack([np.mean(y_train[P == i]) for i in range(k)])

    # Handle clusters with no predictions by assinging them the prediction of their nearst cluster...
    # closest_centroid = np.argpartition(distance.cdist(centroids, centroids, 'euclidean'), 2, axis=1)[:,1]
    # for i in range(k):
    #     if cluster_preds[i] == 0:
    #         print(f' before: {cluster_preds[i]}')
    #         cluster_preds[i] = cluster_preds[closest_centroid[i]]
    #         print(f' after: {cluster_preds[i]}')

    # Find indicies of data closest to each final centroid for test and train
    # data...

    index_final_train = np.argmin(
        distance.cdist(
            x_train,
            centroids,
            'euclidean'),
        axis=1)
    index_final_test = np.argmin(
        distance.cdist(
            x_test,
            centroids,
            'euclidean'),
        axis=1)
    pred_train = np.zeros(len(y_train))
    pred_test = np.zeros(len(y_test))
  
    for i in range(k):
        pred_train[index_final_train == i] = cluster_preds[i]
        pred_test[index_final_test == i] = cluster_preds[i]

    # Calculate RMSE on test data...
    train_RMSE = RMSE(pred_train, y_train)
    test_RMSE = RMSE(pred_test, y_test)
    results = {'Train RMSE': train_RMSE,
               'Test RMSE': test_RMSE,
               'test_preds': pred_test}

    return results
