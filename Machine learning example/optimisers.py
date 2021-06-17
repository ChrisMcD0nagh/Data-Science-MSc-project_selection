from scipy.stats import norm
import numpy as np
import scipy.linalg as la
import time
import random
from algorithms import linear_regression, regression_forest, fit_gaussian, k_means, gaussian_kernel

# Brute force optimiser

def brute_force_optimiser(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        lb,
        ub,
        res=100):

    """Optimise a model with one hyperparameter using brute force."""
    assert len(lb) == 1, "Only 1d hyperparameter optimisation supported."

    hps = np.linspace(lb[0], ub[0], res)
    RMSE_plot = np.zeros((hps.shape[0], 3))
    best_RMSE = np.inf
    best_result_val = None
    best_result_test = None
    opt_hp = None
    for i, hp in enumerate(hps):
        result_val = model(x_train, y_train, x_val, y_val, hp)
        result_test = model(x_train, y_train, x_test, y_test, hp)
        RMSE_plot[i, 0] = result_val['Train RMSE']
        RMSE_plot[i, 1] = result_val['Test RMSE']
        RMSE_plot[i, 2] = result_test['Test RMSE']
        if result_val['Test RMSE'] < best_RMSE:
            best_RMSE = result_val['Test RMSE']
            best_result_val = result_val
            best_result_test = result_test
            opt_hp = hp

    best_result = {}
    best_result['Train RMSE'] = best_result_val['Train RMSE']
    best_result['Val RMSE'] = best_result_val['Test RMSE']
    best_result['Test RMSE'] = best_result_test['Test RMSE']
    best_result['RMSE plot'] = RMSE_plot
    best_result['Optimal HP'] = opt_hp
    best_result['test_preds'] = best_result_test['test_preds']
    if model == linear_regression:
        best_result['coefs'] = best_result_test['coefs']

    return best_result

# Maximum liklihood estimator...

def mle(x, y, steps=100 , lr=1e-1, BO=False):
    """Find the maximum liklihood estimate for the hyper parameters of a Gaussian process."""
    
    # Format y...
    if len(y.shape) > 1:
        y = y[:,0]
    # Generate first guess...
    np.random.seed(42)
    log_l2, log_amp, log_noise = np.random.normal(size=3)
    for _ in range(steps):
        start_step = time.time()
        l2 = np.exp(log_l2)
        amp = np.exp(log_amp)
        noise = np.exp(log_noise)

        K = gaussian_kernel(x,x,l2=l2, gk_var=amp)
        V = K + noise * np.eye(x.shape[0])
        sqdist = np.sum(x**2, 1).reshape(-1, 1) + \
            np.sum(x**2, 1) - 2 * np.dot(x, x.T) 

        # Derivative of V wrt each HP...
        dV_lognoise = noise * np.eye(x.shape[0])
        dV_logamp = K
        dV_logl2 =  sqdist * K / l2**2

        # Evaluate terms for gradients solving a linear system rather than inverting V...
        start_invert = time.time()
        # P, L, U = la.lu(V)
        # solves = [dV_logamp, dV_lognoise, dV_logl2, (y-np.mean(y))]
        # mats = []
        # for s in solves:
        #     z1 = la.solve(P,s)
        #     z2 = la.solve_triangular(L,z1, lower=True)
        #     mats.append(la.solve_triangular(U,z2)) 

        # Vinv_logamp = mats[0]
        # Vinv_lognoise = mats[1]
        # Vinv_logl2 = mats[2]
        # Vinv_y_mean = mats[3]
        # Vinv_logamp = la.solve_triangular(U,la.solve(L, dV_logamp))
        # Vinv_lognoise = la.solve_triangular(U,la.solve(L, dV_lognoise))
        # Vinv_logl2 = la.solve_triangular(U,la.solve(L, dV_logl2))
        # Vinv_y_mean = la.solve_triangular(U,la.solve(L, (y-np.mean(y))))
        # Vinv_logamp = np.linalg.solve(V,dV_logamp)
        # Vinv_lognoise = np.linalg.solve(V,dV_lognoise)
        # Vinv_logl2 = np.linalg.solve(V,dV_logl2)
        # Vinv_y_mean = np.linalg.solve(V,(y-np.mean(y)))
        
        V_inv = np.linalg.inv(V)
        y_mean_Vinv = (y-np.mean(y)).T@V_inv
        Vinv_y_mean = V_inv@(y-np.mean(y))

        end_invert = time.time() - start_invert
        # print(f'Inversion time: {end_invert:.02f}s')

        # Evaluate gradients...
        start_grads = time.time()
        # noise_grad = -0.5*np.trace(Vinv_lognoise)+(0.5*(y-np.mean(y)).T@Vinv_lognoise@Vinv_y_mean)
        # l2_grad = -0.5*np.trace(Vinv_logl2)+(0.5*(y-np.mean(y)).T@Vinv_logl2@Vinv_y_mean)
        # amp_grad = -0.5*np.trace(Vinv_logamp)+(0.5*(y-np.mean(y)).T@Vinv_logamp@Vinv_y_mean)

        noise_grad = -0.5*np.trace(V_inv@dV_lognoise)+(0.5*(y_mean_Vinv@dV_lognoise@Vinv_y_mean))
        l2_grad = -0.5*np.trace(V_inv@dV_logl2)+(0.5*(y_mean_Vinv@dV_logl2@Vinv_y_mean))
        amp_grad = -0.5*np.trace(V_inv@dV_logamp)+(0.5*(y_mean_Vinv@dV_logamp@Vinv_y_mean))
        end_grads = time.time()-start_grads


        # Take next step...
        scaling_fac = 1/x.shape[0]
        log_l2 += lr * scaling_fac * l2_grad
        log_amp += lr * scaling_fac * amp_grad
        log_noise += lr * scaling_fac * noise_grad

        end_step = time.time()-start_step
        if BO != True:
            print(f'Completed step {_+1} in: {end_step:.02f}')
        # ll = -0.5*(np.log(np.linalg.det(V)) + 
        #        (y-np.mean(y)).T@V_inv@(y-np.mean(y)) +
        #        x.shape[0] * np.log(2*np.pi))

    opt_l2 = np.exp(log_l2)
    opt_amp = np.exp(log_amp)
    opt_noise = np.exp(log_noise)
    return [opt_l2, opt_amp, opt_noise]


# Bayesian optimiser (using a gaussain process surrogate model and expected improvement acquisition function)...

def expected_improvement(mu, sigma, opt_y):
    """Acquisition function for Bayesian optimiser. Desgined to give postiive expected improvements when RMSE
    is expected to to be lower than the current best RMSE."""
    e = 1e-10
    z = (opt_y - mu - e) / sigma
    EI = (opt_y - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return EI


def find_next_params(x_known, y_known, opt_y, lbs, ubs, res):
    """Find the optimal next set of hyperparameters to try given the acquistion function."""

    assert (res)**(len(lbs)
                   ) <= 10000, "Resolution too high (intractable) given dimensionality"

    # Standardise x and y training data...
    if x_known.shape[0] > 1:
        x_known_std = (x_known - np.mean(x_known, axis=0)) / \
            np.std(x_known, axis=0)
        y_known_std = (y_known - np.mean(y_known, axis=0)) / \
            np.std(y_known, axis=0)
    else:
        x_known_std = x_known
        y_known_std = y_known
    x_pred_raw = []
    [x_pred_raw.append(np.linspace(lbs[i], ubs[i], res))
     for i in range(len(lbs))]
    x_pred_raw = np.array(x_pred_raw).T

    # Standardize points to predict...
    x_pred_std = (x_pred_raw - np.mean(x_pred_raw, axis=0)) / \
        np.std(x_pred_raw, axis=0)

    # Handle 2 hyperparameters to optimise...
    if len(lbs) == 2:
        xx, yy = np.meshgrid(x_pred_std[:, 0], x_pred_std[:, 1])
        x_pred = np.vstack(
            [np.array((xx[:, i], yy[:, i])).T for i in range(len(xx))])

    # Handle 3 hyperparameters to optimise...
    if len(lbs) == 3:
        xx, yy, zz = np.meshgrid(
            x_pred_std[:, 0], x_pred_std[:, 1], x_pred_std[:, 2])
        x_pred = np.vstack([np.vstack(([np.array(
            (xx[:, i, 0], yy[:, i, 0], zz[:, 0, j])).T for i in range(len(xx))])) for j in range(len(xx))])

    # Use MLE to find optimal HPs for this GP...
    HPs_for_opt = mle(x_known, y_known, steps=100, lr=1e-100, BO=True)

    # Evaluate at each testing point...
    g_fit_start = time.time()
    mu_hp, sigma_hp = fit_gaussian(
        x_known_std, y_known_std, x_pred, None, HPs_for_opt)
    g_fit_end = time.time() - g_fit_start
    # print(f'fit gaussain to HPs time: {g_fit_end}')
    
    # Calculate expected imporvement for each indicies combination...
    EI_vec = expected_improvement(mu_hp[:, 0], sigma_hp, opt_y)
    next_params = x_pred[np.argmax(EI_vec)]

    # Handle if more than one combination gives equal EI...
    if len(next_params.shape) > 1:
        print(f'**Note that mid point seleced for this step as all EIs are equal**')
        next_params = next_params[next_params.shape[0] // 2, :]

    # Unstandartize parameters...
    next_params = next_params * \
        np.std(x_pred_raw, axis=0) + np.mean(x_pred_raw, axis=0)

    return next_params


def GP_Optimiser(model,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 lbs,
                 ubs,
                 res=10,
                 optimisation_steps=5,
                 check_mle = False,
                 kwargs={}):
    """Optimise a model using Bayesian optimisatoin with a Gaussian process surrogate function and expected improvement acquisition function."""
    assert len(lbs) == len(
        ubs), "Lower bound and upper bound dimensions do not match"

    # If optimising GP, check if MLE would be faster...
    if check_mle and res**(3*len(lbs)) > (2*(optimisation_steps-1))*x_train.shape[0]**3/optimisation_steps:
        print("Optimising using MLE...")
        opt_hps_ML = mle(x_train, y_train, steps=optimisation_steps, lr=1e-1)
        results = {'Optimal RMSE': np.nan,
               'Optimal Hyperparameters': opt_hps_ML,
               'All HP steps': [],
               'All RMSEs': []}
        return results
    print("Using Bayesian optimisation...")
    # Standardize x...
    x_train_std = (x_train - np.mean(x_train, axis=0)) / \
            np.std(x_train, axis=0)
    x_test_std = (x_test - np.mean(x_test, axis=0)) / \
            np.std(x_test, axis=0)

    # Ensure y inputs are correctly formatted...
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    # Randomly pick first hyperparameters...
    np.random.seed(42)
    guess1 = []
    [guess1.append(np.random.uniform(lbs[i], ubs[i])) for i in range(len(lbs))]
    guess2 = []
    [guess2.append(np.random.uniform(lbs[i], ubs[i])) for i in range(len(lbs))]

    # Evaluate them...
    RMSE1 = model(
        x_train_std,
        y_train,
        x_test_std,
        y_test,
        guess1,
        rtn_RMSE=True,
        kwargs=kwargs)['Test RMSE']
    RMSE2 = model(
        x_train_std,
        y_train,
        x_test_std,
        y_test,
        guess2,
        rtn_RMSE=True,
        kwargs=kwargs)['Test RMSE']

    # Fit gaussain to hyperparameters and the RMSEs...
    x_known = np.array((guess1, guess2))
    y_known = np.array((RMSE1, RMSE2))
    y_known = y_known.reshape(y_known.shape[0], 1)
    opt_y = min(y_known)

    for _ in range(optimisation_steps):
        start_step = time.time()
        next_params = find_next_params(
            x_known, y_known, opt_y, lbs, ubs, res)

        # Stop here if the next optimal parameters are the same as ones
        # previsouly tested...
        if np.array_equal(next_params, x_known[-1, :]):
            break

        RMSEg1 = model(
            x_train_std,
            y_train,
            x_test_std,
            y_test,
            next_params,
            rtn_RMSE=True,
            kwargs=kwargs)['Test RMSE']

        # Add new HP guesses to x and y...
        x_known_new = np.zeros((x_known.shape[0] + 1, x_known.shape[1]))
        x_known_new[:-1, :] = x_known
        x_known_new[-1, :] = next_params
        x_known = x_known_new
        y_known_new = np.zeros((y_known.shape[0] + 1, y_known.shape[1]))
        y_known_new[:-1, :] = y_known
        y_known_new[-1, :] = RMSEg1
        y_known = y_known_new
        step_end = time.time()-start_step
        print(f'Completed step {_+1} in {step_end:.02f} seconds.')
        # print(f'Total BO step time: {step_end:.02f}')
    
    # Optimal hyperparameters...
    lowest_RMSE = np.amin(y_known)
    optimal_step = np.argwhere(y_known==lowest_RMSE)[0,0] + 1
    optimal_hyperparams = x_known[np.where(y_known == lowest_RMSE)[0], :]

    # Format results...
    results = {'Optimal RMSE': lowest_RMSE,
               'Optimal Hyperparameters': optimal_hyperparams[0],
               'All HP steps': x_known,
               'All RMSEs': y_known,
               'Optimal step': optimal_step}
    return results
