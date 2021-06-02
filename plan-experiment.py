import argparse
import matplotlib.pyplot as plt
import numpy as np
from modules.classes import BinomialExperiment

# Parse command line arguments
# Instantiate parser
parser = argparse.ArgumentParser(description = 'Plan an experiment by providing p_treatment, p_control, desired power level and desired significance threshold')

# Add command line arguments to interpret
parser.add_argument('p_control', type = float, help = 'The expected outcome rate of your control group (status quo outcome rate)')
parser.add_argument('p_treatment', type = float, help = 'The expected outcome rate of your treatment group (target outcome rate of the change you want to test)')
parser.add_argument('--power', type = float, help = 'Optional (default (0.80). Your desired statistical power level.')
parser.add_argument('--alpha', type = float, help = 'Optional (default 0.05). Your desired statistical significance threshold.')

def validate_cmd(args):
    if args.alpha and args.alpha > 0 and args.alpha < 1:
        alpha = args.alpha
    elif args.alpha == None:
        alpha = 0.05
    else:
        raise ValueError('alpha needs to be between 0 and 1.')

    if args.power and args.power > 0 and args.power < 1:
        power = args.power
    elif args.power == None:
        power = 0.80
    else:
        raise ValueError('power needs to be between 0 and 1.')

    if args.p_treatment > 0 and args.p_treatment < 1:
        p_treatment = args.p_treatment
    else:
        raise ValueError('p_treatment needs to be between 0 and 1. Outcome rate in decimal form required.')

    if args.p_control > 0 and args.p_control < 1:
        p_control = args.p_control
    else:
        raise ValueError('p_control needs to be between 0 and 1. Outcome rate in decimal form required.')

    return p_control, p_treatment, power, alpha

def display_figures(figs, nrows = 1, ncols = 1):
    """
    Takes a list of pyplot figures and returns them in the same larger image so that
    all can be viewed together.
    """
    fig, axes = plt.subplots(ncols = ncols, nrows = nrows)
    for i in range(len(figs)):
        axes.ravel()[i].imshow(figs[i], cmap = plt.gray())
        axes.ravel()[i].set_title('Plot {}'.format(i+1))
    plt.tight_layout()

def main():
    """
    Functioning controlling the app's flow. Execute this when this file is run, directly.
    """
    args = parser.parse_args()
    p_control, p_treatment, power, alpha = validate_cmd(args)
    experiment = BinomialExperiment(p_control = p_control, p_treatment = p_treatment, power = power, alpha = alpha)
    fig_p, fig_power, fig_curve = experiment.plan(plot = True, show = True)

if __name__=='__main__':
    main()