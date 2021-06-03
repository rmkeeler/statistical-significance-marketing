import argparse
from PIL import Image
import sys
import os
import webbrowser as wb

from modules.classes import BinomialExperiment
from modules.functions import create_dashboard

# No matter how this script is run, make sure it treats its own directory as the working directory
# This makes sure that relative file referencing always does what's expected
# Keeps all input and output within this project's directory structure
os.chdir(os.path.dirname(sys.argv[0]))


# Parse command line arguments
# Instantiate parser
parser = argparse.ArgumentParser(description = 'Evaluate an experiment after it concludes by providing p_treatment, p_control and sample sizes.')

# Add command line arguments to interpret
parser.add_argument('p_control',
                    type = float,
                    help = 'The expected outcome rate of your control group (status quo outcome rate)')
parser.add_argument('p_treatment',
                    type = float,
                    help = 'The expected outcome rate of your treatment group (target outcome rate of the change you want to test)')
parser.add_argument('n_control',
                    type = int,
                    help = 'Count of observations in your control group.')
parser.add_argument('n_treatment',
                    type = int,
                    help = 'Count of observations in your treatment group.')
parser.add_argument('--show',
                    type = str,
                    help = 'Optional (default no). When yes, output plots to default web browser.')

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

    if args.show and (args.show.lower() in ['yes','no']):
        show = True if args.show.lower() == 'yes' else False
    elif args.show == None:
        show = False
    else:
        raise ValueError('show must be either "yes" or "no" (not case-sensitive)')

    return p_control, p_treatment, n_control, n_treatment, show

def main():
    """
    Function controlling the app's flow. Execute this when this file is run, directly.
    """
    args = parser.parse_args()
    p_control, p_treatment, n_control, n_treatment, show = validate_cmd(args)
    experiment = BinomialExperiment(p_control = p_control,
                                    p_treatment = p_treatment,
                                    n_control = n_control,
                                    n_treatment = n_treatment)
    figs = experiment.evaluate(plot = True)
    if show:
        # Save image to a folder in root called "images" then open them in default image program
        save_location = 'images'
        filename = '/eval.html'

        if not os.path.exists(save_location):
            os.mkdir(save_location)

        create_dashboard(figs, save_location + filename)
        wb.get('chrome %s').open(os.path.dirname(sys.argv[0]) + save_location + filename)

if __name__ == '__main__':
    main()
