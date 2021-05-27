import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def find_power(p_control, p_treatment, n_control = 1000, n_treatment = 1000, alpha = 0.05, report = False, plot = False):
    """
    Takes probabilities observed from a split test and a sample size. Returns
    the statistical power of the significance conclusion.

    This is designed to be a way to check on an experiment in progress or to
    assess a completed experiment that wasn't planned using a methodology like
    that used by find_sample_size() in this module.

    The basic objective of this function is to find the significance threshold
    of a distribution built around the null hypothesis and then find the type II error
    of that same value in a distribution built around the alternative hypothesis. Power
    is derived by inverting that type II error (getting type I error).

    Null: There is no positive difference between the two probabilities observed.
    p_treatment - p_control <= 0

    Alt: There is a positive difference between the two probabilities observed.
    p_treatment - p_control > 0
    """
    # Invert alpha when p_treatment - p_control < 0
    # This allows p_crit calculation to place the critical value in the low tail of the null distribution
    if p_treatment - p_control < 0:
        print('Alt Hypothesis: Treatment - Control < 0\n')
        thresh = 1 - alpha
    else:
        print('Alt Hypothesis: Treatment - Control > 0\n')
        thresh = alpha

    # Get variances of binomial variables based on treatment and control probs
    # REMEMBER: variance of binomial variable is k(p)(1-p), where k is count of trials
    # In a marketing experiment trial count per sample member is 1 (only one chance to click or not per member)
    # In light of hypotheses, we need to consider both control and treatment as parts of the same distribution
    # So, we combine their binomial distributions Control + Treatment
    # Variances are likewise added Var(Treatment) + Var(Control)
    variance_control = (1 * p_control * (1 - p_control))
    variance_treatment = (1 * p_treatment * (1 - p_treatment))

    # Get standard error of null distribution
    # Using control for both elements of calculation, because null hypothesis is that Treatment + Control is same as Control + Control
    # Both treatment and control are parts of equivalent binomial distributions
    sterror_null = np.sqrt((variance_control / n_control) + (variance_control / n_treatment))
    # Get standard error of alt distribution (assumes treatment - prob is mean of its own separate binomial variable)
    sterror_alt = np.sqrt((variance_control / n_control) + (variance_treatment / n_treatment))

    dist_null = stats.norm(loc = 0, scale = sterror_null)
    dist_alt = stats.norm(loc = p_treatment - p_control, scale = sterror_alt)

    p_crit = dist_null.ppf(1 - thresh) # ppf is percent point function. Finds value at percentile of value provided (95th percentile when alpha is 0.05)
    beta = dist_alt.cdf(p_crit) # cdf is cumulative distribution function. Finds % of values in dist below value provided.

    power = (1 - beta) if (p_treatment > p_control) else beta

    if report:
        print('Control Sample Size: {:,}\nTreatment Sample Size: {:,}'.format(n_control, n_treatment))
        print('Significance Threshold: {:.4f}'.format(alpha))
        print('Statisical Power: {:.4f}'.format(power))

    if plot:
        sample_null = dist_null.rvs(size = n_control)
        sample_alt = dist_alt.rvs(size = n_treatment)

        lowest_x = min(min(sample_null), min(sample_alt))
        highest_x = max(max(sample_null), max(sample_alt))

        x = np.linspace(lowest_x, highest_x, 1000)

        y_null = dist_null.pdf(x)
        y_alt = dist_alt.pdf(x)

        color_null = 'blue'
        color_alt = 'orange'

        fig, ax = plt.subplots(1,1,figsize = (8,6))
        ax.plot(x, y_null, color = color_null)
        ax.plot(x, y_alt, color = color_alt)
        ax.vlines(x = p_crit, ymin = 0, ymax = max([dist_null.pdf(p_crit), dist_alt.pdf(p_crit)]), linestyle = 'dashed', color = 'black')
        if p_treatment >= p_control:
            ax.fill_between(x, y_alt, 0, where = (x <= p_crit), color = color_alt, alpha = 0.5)
            ax.fill_between(x, y_null, 0, where = (x >= p_crit), color = color_null, alpha = 0.5)
        else:
            ax.fill_between(x, y_alt, 0, where = (x >= p_crit), color = color_alt, alpha = 0.5)
            ax.fill_between(x, y_null, 0, where = (x <= p_crit), color = color_null, alpha = 0.5)
        ax.legend(['Null','Alt'])

        plt.show();

    return power


def find_sample_size(p_control, p_treatment, alpha = 0.05, power = 0.8, report = False):
    """
    Takes a desired significance threshold (alpha) and statistical power,
    then figures out what sample size is necessary to detect a desired
    increase in probability from an experiment.

    Null: p_treatment - p_control <= 0
    Alt: p_treatment - p_control > 0

    One-tailed test. Useful when running experiments to answer the question
    "will this idea improve results?"

    Returns size of a single group in the full experiment sample. Full A/B test
    audience should be sample_size * 2.
    """
    if power == 1:
        print("NOTICE: Sample size is infinity as power approaches 1. Using 0.99 as power, instead.\n")
        power = 0.99

    if power == 0:
        print("NOTICE: Sample size is undefined when power is 0. Using 0.01 as power, instead.\n")
        power = 0.01

    z_null = stats.norm.ppf(1-alpha)
    z_alt = stats.norm.ppf(1-power)

    # stdev is just sqrt(variance). Variance of difference of random vars is sum of both variances.
    # Variance of binomial random variable is k(p)(1-p), and k trials is 1 in a marketing experiment (click or don't, one chance)
    # First, null hypothesis is no difference between treatment_p and control_p. Part of same distribution
    stdev_null = np.sqrt((p_control * (1 - p_control)) + (p_control * (1 - p_control)))
    # Second, alt is that p_treatment - p_null > 0. Subtracting distributions means adding their variances.
    stdev_alt = np.sqrt((p_treatment * (1 - p_treatment)) + (p_control * (1 - p_control)))

    z_diff = (z_null*stdev_null) - (z_alt * stdev_alt) # null - alt due to the relatinship between stdev, standard error and sample size
    p_diff = p_treatment - p_control # alt - null, because this is the difference we're actually testing. intuitive.

    n = (z_diff / p_diff) ** 2

    sample_size = int(np.ceil(n))

    if report:
        print('Minimal Practical Difference Needed: {:.4f}'.format(p_treatment - p_control))
        print('Desired Statistical Power: {}'.format(power))
        print('Recommended Responses per Sample Group: {:,}'.format(sample_size))

    return n
