import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

class BinomialExperiment():
    """
    Creates an object that represents observed or desired split test results.
    Currently only supports two-way split tests. n-way tests will be supported
    in a future release.

    Analyses can then be performed on this object by calling this class's methods.
    Return statistical power, estimate a necessary sample size, return statistical
    significance. Plotting is also possible.

    Marketing program/campaign optimizaiton is the intended use case. Therefore,
    split tests are evaluated with one-way significance tests and under these hypotheses:

    Null: Treatment Probability - Control Probability <= 0
    Alt: Treatment Probability - Control Probability > 0

    Also, this class is designed to be used as the backend of a web application
    that helps marketers plan and understand optimization experiments.
    """
    def __init__(self, p_control = None, p_treatment = None, n_control = None, n_treatment = None, power = None, alpha = 0.05):
        """
        Only two required args are p_control and p_treatment. It is assumed that the user is either evaluating a completed
        experiment or has already determined the practical difference necessary to make an experiment's results worthwhile.

        So, those two values are already on-hand.
        """
        self.p_control = p_control
        self.p_treatment = p_treatment

        self.n_control = n_control
        self.n_treatment = n_treatment

        self.var_control = 1 * p_control * (1 - p_control)
        self.var_treatment = 1 * p_treatment * (1 - p_treatment)

        self.sim_null = None
        self.sim_alt = None

        if power == 1:
            print('Sample size approaches infinity as power approaches 1, so 1 is an invalid power vlaue. Changing power to 0.99.')
            self.power = 0.99
        elif power == 0:
            print('Sample size is undefined at power of 0. Changing power to 0.01.')
            self.power = 0.01
        else:
            self.power = power

        self.alpha = alpha
        self.p_value = None

    def get_p_sample(self):
        """
        Take sample sizes and probabilities from each sample and return the probability of the combination of samples
        """
        control = self.p_control * self.n_control
        treatment = self.p_treatment * self.n_treatment
        sample = self.n_control + self.n_treatment

        p_sample = (control + treatment) / sample

        self.p_sample = p_sample

        return p_sample

    def estimate_sample(self):
        """
        Take desired effect size, alpha and desired power level. Return a minimum sample size (one group)
        that would be necessary to acheive the desired experiment results.

        Allows the user to specify power and alpha here, if they didn't specify them when they instantiated the class.
        Otherwise, it takes the values provided to the class on instantiation.
        """
        z_null = stats.norm.ppf(1 - self.alpha)
        z_alt = stats.norm.ppf(1 - self.power)

        stdev_null = np.sqrt(self.var_control + self.var_control)
        stdev_alt = np.sqrt(self.var_control + self.var_treatment)

        z_diff = (z_null * stdev_null) - (z_alt * stdev_alt)
        p_diff = self.p_treatment - self.p_control

        n = (z_diff / p_diff) ** 2

        sample_size = int(np.ceil(n))

        self.n_control = sample_size
        self.n_treatment = sample_size

        return sample_size

    def analyze_significance(self):
        """
        Take sample sizes and probabilities and return the significance of the difference between the probabilities.
        One-tailed test.

        Null: Treatment Prob - Control Prob <= 0
        Alt: Treatment Prob - Control Prob > 0
        """
        var_control = 1 * self.p_sample * (1 - self.p_sample)
        var_treatment = 1 * self.p_sample * (1 - self.p_sample) # Same as var_control, because null hyp is no difference

        sigma = np.sqrt((var_control / self.n_control) + (var_treatment / self.n_treatment))

        z = (self.p_treatment - self.p_control) / sigma
        p = (1 - stats.norm.cdf(z))
        self.p_value = p

        return p

    def simulate_significance(self):
        """
        Same intent and outcome as analyze_significance(), but it simulates a binomial distribution rather than
        approximating one with a normal distribution. No continuity correction, necessary. Only significant source of
        inaccuracy would be variability between runs (random simulations can yield slightly different outcomes, each time).
        """
        observed_difference = self.p_treatment - self.p_control

        rng = np.random.default_rng()
        sample_control = rng.binomial(n = self.n_control, p = self.p_sample, size = 1000000) / self.n_control
        sample_treatment = rng.binomial(n = self.n_treatment, p = self.p_sample, size = 1000000) / self.n_treatment

        differences = sample_treatment - sample_control

        p = (differences >= observed_difference).mean()
        self.p_value = p

        return p

    def simulate_power(self):
        """
        Takes results of a completed experiment and reveals the statistical power of the significance conclusion.
        """
        if self.p_treatment - self.p_control < 0:
            thresh = 1 - self.alpha
        else:
            thresh = self.alpha

        sterror_null = np.sqrt((self.var_control / self.n_control) + (self.var_control / self.n_control))
        sterror_alt =  np.sqrt((self.var_treatment / self.n_treatment) + (self.var_control / self.n_control))
        self.sterror_null = sterror_null
        self.sterror_alt = sterror_alt

        dist_null = stats.norm(loc = 0, scale = sterror_null)
        dist_alt = stats.norm(loc = self.p_treatment - self.p_control, scale = sterror_alt)
        self.sim_null = dist_null
        self.sim_alt = dist_alt

        p_crit = dist_null.ppf(1 - thresh)
        beta = dist_alt.cdf(p_crit)

        power = (1 - beta) if self.p_treatment > self.p_control else beta
        self.power = power

        return power

    def plot_p(self, show = False):
        """
        Plot the null distribution, treatment probability and then shade the p value in order to visualize the results
        of a significance test.
        """
        rng = np.random.default_rng()

        control_sample = rng.binomial(n = self.n_control, p = self.p_sample, size = 1000000) / self.n_control
        treatment_sample = rng.binomial(n = self.n_treatment, p = self.p_sample, size = 1000000) / self.n_treatment

        difference = treatment_sample - control_sample
        observed_difference = self.p_treatment - self.p_control

        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(1,1,1)

        mu, sigma = stats.norm.fit(difference)
        crit_density = stats.norm.pdf(observed_difference, mu, sigma)

        x = np.linspace(min(difference), max(difference), self.n_control + self.n_treatment)
        y = stats.norm.pdf(x, mu, sigma)

        ax.plot(x, y, 'blue')
        ax.vlines(x = observed_difference, ymin = 0, ymax = crit_density, linestyle = 'dashed', color = 'black')
        ax.fill_between(x, y, 0, where = (x >= observed_difference), color = 'green', alpha = 0.5)

        ax.set_xlabel('Difference in Probabilities')
        ax.set_ylabel('Density')

        if show:
            plt.show();

        return fig, ax

    def plot_power(self, show = False):
        """
        Produce a plot demonstrating the statistical power of the binomial split test's results.
        Plots a simulated null distribution, a simulated alt distribution, the critical value of the null distribution.

        Null p and alt beta are shaded to convey power in shorthand.

        Need to call .simulate_power() before this to populate power, sim_null and sim_alt attributes.
        """
        if self.p_treatment - self.p_control < 0:
            thresh = 1 - self.alpha
        else:
            thresh = self.alpha

        p_crit = self.sim_null.ppf(1 - thresh)
        beta = self.sim_alt.cdf(p_crit)

        sample_null = self.sim_null.rvs(size = self.n_control)
        sample_alt = self.sim_alt.rvs(size = self.n_treatment)

        lowest_x = min(min(sample_null), min(sample_alt))
        highest_x = max(max(sample_null), max(sample_alt))

        x = np.linspace(lowest_x, highest_x, self.n_control + self.n_treatment)

        y_null = self.sim_null.pdf(x)
        y_alt = self.sim_alt.pdf(x)

        color_null = 'blue'
        color_alt = 'orange'

        fig = plt.figure(figsize = (8,6))
        ax = fig.add_subplot(1,1,1)

        ax.plot(x, y_null, color = color_null)
        ax.plot(x, y_alt, color = color_alt)
        ax.vlines(x = p_crit, ymin = 0, ymax = max([self.sim_null.pdf(p_crit), self.sim_alt.pdf(p_crit)]), linestyle = 'dashed', color = 'black')
        if self.p_treatment >= self.p_control:
            ax.fill_between(x, y_alt, 0, where = (x <= p_crit), color = color_alt, alpha = 0.5)
            ax.fill_between(x, y_null, 0, where = (x >= p_crit), color = color_null, alpha = 0.5)
        else:
            ax.fill_between(x, y_alt, 0, where = (x >= p_crit), color = color_alt, alpha = 0.5)
            ax.fill_between(x, y_null, 0, where = (x <= p_crit), color = color_null, alpha = 0.5)

        ax.set_xlabel('Sample Mean Differences (Probabilities)')
        ax.set_ylabel('Probability Density')
        ax.legend(['Null','Alt'])

        if show:
            plt.show();

        return fig, ax

    def evaluate(self, plot = False, show = False):
        """
        Calls other methods in this class in order to speed up the experiment evaluation
        process and make this class more intuitive to use.

        User can treat this class as a container for parameters of an experiment that has
        concluded. Calling evaluate on it will generate P, Power and some Plots if plot == True.

        Will call plt.show(); on each plot, if show == True.
        """
        self.sample_p = self.get_p_sample()
        self.p_value = self.simulate_significance()
        self.power = self.simulate_power()

        print(self)

        if plot:
            fig1, ax1 = self.plot_p(show = show)
            fig2, ax2 = self.plot_power(show = show)

            return [fig1, ax1], [fig2, ax2]

    def plan(self):
        """
        Call other methods in this class in order to speed up the experiment planning
        flow and make this class more intuitive to use.

        When planning an experiment, user will leave n_control and n_treatment blank.
        This is because the objective there is figuring out how many observations to generate
        in order to detect a minimum practical effect size at a reasonable degree of power.

        Before calling this method, user has populated p_control, p_treatment, power and alpha.
        Power is desired power level. p_control is status quo rate. p_treatment is minimum outcome
        rate required to be meaningful to the business. Alpha is desired significance level
        (almost always, 0.05 is desired).
        """
        self.n_control = self.estimate_sample()
        self.n_treatment = self.n_control
        self.sample_p = self.get_p_sample()
        self.p_value = self.simulate_significance()

        print(self)

    def __repr__(self):
        """
        Magic method that outputs the experiment's parameters, so far.
        """
        header = '|||Experiment Readout|||\n'
        data = [['Control Probability', '{:.2%}'.format(self.p_control)],
               ['Treatment Probability', '{:.2%}'.format(self.p_treatment)],
               ['Effect Size', '{:.2%}'.format(self.p_treatment - self.p_control)],
               ['',''],
               ['Control Sample Size', '{:,}'.format(self.n_control)],
               ['Treatment Sample Size', '{:,}'.format(self.n_treatment)],
               ['',''],
               ['Statistical Power', '{:.3f}'.format(self.power)],
               ['Significance Threshold', '{:.3f}'.format(self.alpha)],
               ['P Value', '{:.3f}'.format(self.p_value) if self.p_value else 'None']]

        return header + str(pd.DataFrame(data = [x[1] for x in data], index = [x[0] for x in data], columns = ['']))
