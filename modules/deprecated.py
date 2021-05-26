import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def dep_sim_sig_test_a(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, runs = 200000, plot = False):
    """
    NOTE: Deprecated. Much faster to use np.random.binomial() for this.
    Use the version without "dep_" in front of its name.
    Keeping this in the module to remind my future self of the detailed steps.

    Get a p value from split test results. Two-tailed assessment of treatment's
    difference from control's value.

    Null: Treatment is not different than control.
    Alt: Treatment is different than control.

    Implicit assumption: Control's mean is the true population mean without treatment.
    """
    # This list will become our null distribution, later.
    sample_means = []

    for i in range(runs):
        # Simulate a bunch of datasets same size as control group
        # Add means of these simulations to sample_means to create null distribution of means
        # Null distribution based on control mean, because of our implicit assumption (see docstring)
        sample = np.random.choice([0,1], size = control_size, replace = True, p = [1-control_prob, control_prob])
        sample_means.append(np.mean(sample))

    # Sampling mean will serve as our contrast with the observed treatment mean when we calc p value
    sampling_mean = np.mean(sample_means)

    sample_means = np.asarray(sample_means)
    mean_diff = treatment_prob - sampling_mean

    # extreme_high and extreme_low let this script adapt to cases when treatment performs worse than control
    extreme_observed = treatment_prob
    extreme_inverse = sampling_mean - mean_diff
    extreme_high = max(extreme_observed, extreme_inverse)
    extreme_low = min(extreme_observed, extreme_inverse)

    print('Max extreme: {}'.format(extreme_high))
    print('Min extreme: {}'.format(extreme_low))

    # This is the part where we get p value.
    # % of instances in null distribution that are equal to or more extreme than our observed treatement mean
    proportion_upper = (sample_means >= extreme_high).mean()
    proportion_lower = (sample_means <= extreme_low).mean()

    p = proportion_upper + proportion_lower
    print('P Value: {}'.format(p))

    if plot:
        fig, ax = plt.subplots(1,1,figsize=(8,6))

        ax.hist([w for w in sample_means if w < extreme_low], color = 'red', bins = 5)
        ax.hist([w for w in sample_means if w > extreme_high], color = 'red', bins = 5)
        ax.hist([w for w in sample_means if (w > extreme_low) and (w < extreme_high)], color = 'grey', bins = 10)
        ax.axvline(x = extreme_observed, linestyle = 'dashed', color = 'red')
        ax.axvline(x = extreme_inverse, linestyle = 'dashed', color = 'red')

        plt.show();

    return p

def dep_sim_sig_test_b(control_prob, treatment_prob, tails = 1, control_size = 1000, treatment_size = 1000, runs = 200000, plot = False):
    """
    NOTE: Deprecated. Much faster to use np.random.binomial() for this.
    Use the version without "dep_" in front of its name.
    Keeping this in the module to remind my future self of the detailed steps.

    Test for significance using the method in part 2, above. Default 1-tailed.

    Null: Difference treatment - control is 0 or less.
    Alt: Difference treatment - control is positive.

    Implicit assumption is that treatment and control are the same where the eval metric is concerned.
    The precise values obtained for each prob are less important.
    Greater external validity with this method in marketing program optimization contexts.
    """
    sample_prob = ((control_prob * control_size) + (treatment_prob * treatment_size)) / (control_size + treatment_size)
    print('Full Sample Probability: {:.4f}'.format(sample_prob))

    sample_means = []

    for i in range(runs):
        sample_control = np.random.choice(a = [0,1], size = control_size, p = [1-sample_prob, sample_prob])
        sample_treatment = np.random.choice(a = [0,1], size = treatment_size, p = [1-sample_prob, sample_prob])

        mean_difference = np.mean(sample_control) - np.mean(sample_treatment)

        sample_means.append(mean_difference)

    sample_means = np.asarray(sample_means)

    null_difference = np.mean(sample_means)
    observed_difference = treatment_prob - control_prob

    proportion_extreme = (sample_means >= observed_difference).mean()

    p = proportion_extreme
    print('P Value: {}'.format(p))

    if plot:
        fig, ax = plt.subplots(1,1,figsize = (8,6))

        ax.hist(x = [w for w in sample_means if w < observed_difference], color = 'grey', bins = 10)
        ax.hist(x = [w for w in sample_means if w >= observed_difference], color = 'red', bins = 5)
        ax.axvline(x = observed_difference, linestyle = 'dashed', color = 'red')

        plt.show();

    return p
