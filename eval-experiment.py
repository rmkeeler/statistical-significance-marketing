import argparse
import numpy as np
from modules.classes import BinomialExperiment

# Parse command line arguments
# Instantiate parser
parser = argparse.ArgumentParser(description = 'Evaluate an experiment after it concludes by providing p_treatment, p_control and sample sizes.')

# Add command line arguments to interpret
parser.add_argument('p_control', type = float, help = '')
parser.add_argument('p_treatment', type = float, help = '')
parser.add_argument('n_control', type = int, help = '')
parser.add_argument('n_treatment', type = int, help = '')

def validate_cmd(args):
    """
    Check each argument to make sure values are appropriate for this analysis.
    """
    if args.p_control > 0 and args.p_control < 1:
        p_control = args.p_control
    else:
        raise ValueError('Invalid p_control. p_control needs to be between 0 and 1. Is {}'.format(args.p_control))

    if args.p_treatment > 0 and args.p_treatment < 1:
        p_treatment = args.p_treatment
    else:
        raise ValueError('Invalid p_treatment. p_treatment needs to be between 0 and 1. Is {}'.format(args.p_treatment))

    if args.n_control == int(args.n_control) and args.n_treatment == int(args.n_treatment):
        n_control = args.n_control
        n_treatment = args.n_treatment
    else:
        raise ValueError('n_control and n_treatment must both be ints for this analysis to work.')

    return p_control, p_treatment, n_control, n_treatment

def main():
    """
    Function controlling the app's flow. Execute this when this file is run, directly.
    """
    args = parser.parse_args()
    p_control, p_treatment, n_control, n_treatment = validate_cmd(args)
    experiment = BinomialExperiment(p_control = p_control,
                                    p_treatment = p_treatment,
                                    n_control = n_control,
                                    n_treatment = n_treatment)
    fig_p, fig_power = experiment.evaluate(plot = True, show = True)

if __name__ == '__main__':
    main()
