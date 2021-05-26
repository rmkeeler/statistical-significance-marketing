import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def find_power():
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
