import numpy as np
import scipy.stats as stats
import pandas as pd

import os

import plotly.graph_objects as go

class NormalExperiment():
    """
    Used to plan and evaluate experiments that contrast averages (rather than
    proportions).

    Analyses can then be performed on this object by calling this class's methods.
    Return statistical power, estimate a necessary sample size, return statistical
    significance. Plotting is also possible.

    Marketing program/campaign optimizaiton is the intended use case. Therefore,
    split tests are evaluated with one-way significance tests and under these hypotheses:

    Null: Treatment Mean - Control Mean <= 0
    Alt: Treatment Mean - Control Mean > 0

    Also, this class is designed to be used as the backend of a web application
    that helps marketers plan and understand optimization experiments.
    """
    def __init__(self, u_control = 0, u_treatment = 0, n_control = 0, n_treatment = 0, power = None, alpha = 0.05):
        """
        Only two required args are u_control and u_treatment (the means).

        It is assumed that the user is either evaluating a completed experiment
        or has already determined the practical difference necessary to make an
        experiment's results worthwhile.

        So, those two values are already on-hand.
        """
        self.u_control = u_control
        self.u_treatment = u_treatment

        self.n_control = n_control
        self.n_treatment = n_treatment

        self.norm_null = None
        self.norm_alt = None

        self.confidence_control = None
        self.confidence_treatment = None

        if n_control > 0 and n_treatment > 0 and u_control > 0 and u_treatment > 0:
            control = self.u_control * self.n_control
            treatment = self.u_treatment * self.n_treatment
            sample = self.n_treatment + self.n_control

            self.u_sample = (control + treatment) / sample
        else:
            self.u_sample = None

        if power == 1:
            print('Sample size approaches infinity as power approaches 1, so 1 is an invalid power vlaue. Changing power to 0.99.')
            self.power = 0.99
        elif power == 0:
            print('Sample size is undefined at power of 0, so 0 is an invalid power value. Changing power to 0.01.')
            self.power = 0.01
        else:
            self.power = power

        self.alpha = alpha
        self.p_value = None
