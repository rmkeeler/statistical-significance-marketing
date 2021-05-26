import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def sim_sig_test_a(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, runs = 500000, plot = False):
    """
    Get a p value from split test results. Two-tailed assessment of treatment's
    difference from control's value. Uses control probability as the basis of the
    evaluation.

    Evaluates via simulation of samples drawn [runs] times rather than analytically.

    Null: Treatment is not different than control.
    Alt: Treatment is different than control.

    Implicit assumption: Control's mean is the true population mean without treatment.
    """
    rng = np.random.default_rng()
    sample = rng.binomial(n = control_size, p = control_prob, size = runs) / control_size

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
    print('Observed Difference: {:.4f}'.format(observed_difference))

    # Create binomial distribution. Normalize for sample size to get a distribution of outcome rates
    # Rather than counts of outcomes.
    rng = np.random.default_rng()
    sample_control = rng.binomial(n = control_size, p = sample_prob, size = runs) / control_size
    sample_treatment = rng.binomial(n = treatment_size, p = sample_prob, size = runs) / treatment_size

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
