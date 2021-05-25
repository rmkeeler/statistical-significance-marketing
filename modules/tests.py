import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_sample_prob(control_prob, treatment_prob, control_size, treatment_size):
    """
    When all we have are control and treatment probabilities and control and
    treatment sizes, this function is a convenient way to get an overall sample
    probability.

    Intended to be used to simplify programming for significance tests for which
    null hypothesis is that control prob and treatment prob are the same (difference is 0)
    """
    controls = control_prob * control_size
    treatments = treatment_prob * treatment_size
    sample_size = control_size + treatment_size

    sample_prob = (controls + treatments) / sample_size

    return sample_prob

def sim_sig_test_a(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, runs = 200000, plot = False):
    """
    Get a p value from split test results. Two-tailed assessment of treatment's
    difference from control's value. Uses control probability as the basis of the
    evaluation.

    Null: Treatment is not different than control.
    Alt: Treatment is different than control.

    Implicit assumption: Control's mean is the true population mean without treatment.
    """
    sample = np.random.binomial(n = control_size, p = control_prob, size = runs) / control_size

    diff = treatment_prob - np.mean(sample)

    extremity_observed = treatment_prob
    extremity_inverted = np.mean(sample) - diff

    lowerbound = min(extremity_observed, extremity_inverted)
    upperbound = max(extremity_observed, extremity_inverted)

    p = (sample <= lowerbound).mean() + (sample >= upperbound).mean()
    print('P Value: {}'.format(p))

    if plot:
        fig, ax = plt.subplots(1,1,figsize = (8,6))
        mu, sigma = stats.norm.fit(sample)

        crit_density_low = stats.norm.pdf(lowerbound, mu, sigma)
        crit_density_high = stats.norm.pdf(upperbound, mu, sigma)

        x = np.linspace(min(sample), max(sample), control_size)
        y = stats.norm.pdf(x, mu, sigma)

        ax.plot(x, y, color = 'blue')
        ax.vlines(x = lowerbound, ymin = 0, ymax = crit_density_low, color = 'black', linestyle = 'dashed')
        ax.vlines(x = upperbound, ymin = 0, ymax = crit_density_high, color = 'black', linestyle = 'dashed')
        ax.fill_between(x, y, 0, where = (x >= upperbound), color = 'green', alpha = 0.5)
        ax.fill_between(x, y, 0, where = (x <= lowerbound), color = 'green', alpha = 0.5)

        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')

        plt.show();

    return p

def sim_sig_test_b(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, runs = 500000, plot = False):
    """
    Get a p value from split test results. One-tailed assessment of difference
    treatment - control. Uses difference == 0 as the basis for evaluation.

    Null: Difference treatment - control is 0 or less.
    Alt: Difference treatment - control is positive.

    Implicit assumption is that treatment and control are the same where the eval metric is concerned.
    The precise values obtained for each prob are less important.
    Greater external validity with this method in marketing program optimization contexts.
    """
    sample_size = treatment_size + control_size
    sample_prob = get_sample_prob(control_prob, treatment_prob, control_size, treatment_size)

    observed_difference = treatment_prob - control_prob
    print('Observed Difference: {}'.format(observed_difference))

    # Create binomial distribution. Normalize for sample size to get a distribution of outcome rates
    # Rather than counts of outcomes.
    sample_control = np.random.binomial(n = control_size, p = sample_prob, size = runs) / control_size
    sample_treatment = np.random.binomial(n = treatment_size, p = sample_prob, size = runs) / treatment_size

    differences = sample_treatment - sample_control

    p = (differences >= observed_difference).mean()
    print('P Value: {}'.format(p))

    if plot:
        #plt.hist(differences);
        fig, ax = plt.subplots(1,1,figsize = (8,6))
        mu, sigma = stats.norm.fit(differences)
        crit_density = stats.norm.pdf(observed_difference, mu, sigma)

        x = np.linspace(min(differences),max(differences),sample_size)
        y = stats.norm.pdf(x, mu, sigma)

        ax.plot(x, y, color = 'blue')
        ax.vlines(x = observed_difference, linestyle = 'dashed', color = 'black', ymin = 0, ymax = crit_density)
        # Shade the chart between line (x, y) and axis (0) for x above or equal to our crit value
        ax.fill_between(x, y, 0, where = (x >= observed_difference), alpha = 0.5, color = 'green')

        ax.set_xlabel('Difference in Probabilities')
        ax.set_ylabel('Density')

        plt.show();

    return p

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
