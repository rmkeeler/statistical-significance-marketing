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

def analytic_a(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, plot = False):
    """
    Get a p value from split test results. Two-tailed assessment of treatment's
    difference from control's value. Uses control probability as the basis of the
    evaluation.

    Evaluates analytically rather than via simulation.

    Null: Treatment is not different than control.
    Alt: Treatment is different than control.

    Implicit assumption: Control's mean is the true population mean without treatment.
    """
    variance_control = 1 * control_prob * (1 - control_prob)
    sigma = np.sqrt(variance_control/control_size)

    z = (treatment_prob - control_prob) / sigma
    p = 2 * (1 - stats.norm.cdf(z)) if treatment_prob >= control_prob else 2 * (stats.norm.cdf(z))

    print('Z Score: {}\nP Value: {}'.format(z,p))

    if plot:
        rng = np.random.default_rng()
        sample = rng.binomial(p = control_prob, n = control_size, size = 500000) / control_size

        diff = treatment_prob - np.mean(sample)

        extremity_observed = treatment_prob
        extremity_inverted = np.mean(sample) - diff

        lowerbound = min(extremity_observed, extremity_inverted)
        upperbound = max(extremity_observed, extremity_inverted)

        fig, ax = plt.subplots(1,1,figsize = (8,6))
        mu, sigma = stats.norm.fit(sample)

        crit_density_high = stats.norm.pdf(upperbound, mu, sigma)
        crit_density_low = stats.norm.pdf(lowerbound, mu, sigma)

        x = np.linspace(min(sample), max(sample), control_size)
        y = stats.norm.pdf(x, mu, sigma)

        ax.plot(x, y, color = 'blue')
        ax.vlines(x = lowerbound, ymin = 0, ymax = crit_density_low, color = 'black', linestyle = 'dashed')
        ax.vlines(x = upperbound, ymin = 0, ymax = crit_density_high, color = 'black', linestyle = 'dashed')
        ax.fill_between(x, y, 0, where = (x <= lowerbound), color = 'green', alpha = 0.5)
        ax.fill_between(x, y, 0, where = (x >= upperbound), color = 'green', alpha = 0.5)

        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')

        plt.show();

        return fig, ax, p
    else:
        return p

def analytic_b(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, plot = False):
    """
    Get a p value from split test results. One-tailed assessment of difference
    treatment - control. Uses difference == 0 as the basis for evaluation.

    Evaluates analytically rather than via simulation.

    Null: Difference treatment - control is 0 or less.
    Alt: Difference treatment - control is positive.

    Implicit assumption is that treatment and control are the same where the eval metric is concerned.
    The precise values obtained for each prob are less important.
    Greater external validity with this method in marketing program optimization contexts.
    """
    sample_prob = get_sample_prob(control_prob, treatment_prob, control_size, treatment_size)

    variance_control = 1 * sample_prob * (1 - sample_prob)
    variance_treatment = 1 * sample_prob * (1 - sample_prob) # same as variance_control, because null assumes probs are equivalent
    sigma = np.sqrt((variance_control/control_size) + (variance_treatment/treatment_size))

    z = (treatment_prob - control_prob) / sigma
    p = (1 - stats.norm.cdf(z))

    print('Z Score: {}\nP Value: {}'.format(z,p))

    if plot:
        rng = np.random.default_rng()
        control_sample = rng.binomial(p = sample_prob, n = control_size, size = 500000) / control_size
        treatment_sample = rng.binomial(p = sample_prob, n = treatment_size, size = 500000) / treatment_size

        differences = treatment_sample - control_sample
        observed_difference = treatment_prob - control_prob

        fig, ax = plt.subplots(1,1,figsize = (8,6))
        mu, sigma = stats.norm.fit(differences)
        crit_density = stats.norm.pdf(observed_difference, mu, sigma)

        x = np.linspace(min(differences), max(differences), control_size + treatment_size)
        y = stats.norm.pdf(x, mu, sigma)

        ax.plot(x, y, color = 'blue')
        ax.vlines(x = observed_difference, ymin = 0, ymax = crit_density, linestyle = 'dashed', color = 'black')
        ax.fill_between(x, y, 0, where = (x >= observed_difference), color = 'green', alpha = 0.5)

        ax.set_xlabel('Difference in Probabilities')
        ax.set_ylabel('Density')

        plt.show();

        return fig, ax, p
    else:
        return p

def simulation_a(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, runs = 500000, plot = False):
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

        return fig, ax, p

    else:
        return p

def simulation_b(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, runs = 500000, plot = False):
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
    sample_prob = calcs.get_sample_prob(control_prob, treatment_prob, control_size, treatment_size)

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

        return fig, ax, p

    else:
        return p
