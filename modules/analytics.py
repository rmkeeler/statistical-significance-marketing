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

def analytic_sig_test_a(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, plot = False):
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
    p = 2 * (1 - stats.norm.cdf(z))

    print('Z Score: {}\nP Value: {}'.format(z,p))

    return p

def analytic_sig_test_b(control_prob, treatment_prob, control_size = 1000, treatment_size = 1000, plot = False):
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
    variance_treatment = 1 * sample_prob * (1 - sample_prob) # same as variance_control, because we null assumes probs are equivalent
    sigma = np.sqrt(variance_control/control_size + variance_treatment/treatment_size)

    z = (treatment_prob - control_prob) / sigma
    p = (1 - stats.norm.cdf(z))

    print('Z Score: {}\nP Value: {}'.format(z,p))

    return p
