import argparse
import os
import sys
from PIL import Image
import webbrowser

from modules.classes import BinomialExperiment
from modules.functions import create_dashboard
from modules.functions import save_images

os.chdir(os.path.dirname(sys.argv[0]))

# Parse command line arguments
# Instantiate parser
parser = argparse.ArgumentParser(description = 'Plan an experiment by providing p_treatment, p_control, desired power level and desired significance threshold')

# Add command line arguments to interpret
parser.add_argument('p_control',
                    type = float,
                    help = 'The expected outcome rate of your control group (status quo outcome rate)')
parser.add_argument('p_treatment',
                    type = float,
                    help = 'The expected outcome rate of your treatment group (target outcome rate of the change you want to test)')
parser.add_argument('--power',
                    type = float,
                    help = 'Optional (default (0.80). Your desired statistical power level.')
parser.add_argument('--alpha',
                    type = float,
                    help = 'Optional (default 0.05). Your desired statistical significance threshold.')
parser.add_argument('--show',
                    type = str,
                    help = 'Optional (default no). Yes will generate plots in your default web browser. No skips that step.')

def validate_cmd(args):
    """
    Check each argument to make sure values are appropriate for this analysis.
    """
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

    if args.show and (args.show.lower() in ['yes','no']):
        show = True if args.show.lower() == 'yes' else False
    elif args.show == None:
        show = False
    else:
        raise ValueError('show needs to be "yes" or "no" (not case-sensitive)')

    return p_control, p_treatment, power, alpha, show

def main():
    """
    Function controlling the app's flow. Execute this when this file is run, directly.
    """
    args = parser.parse_args()
    p_control, p_treatment, power, alpha, show = validate_cmd(args)
    experiment = BinomialExperiment(p_control = p_control,
                                    p_treatment = p_treatment,
                                    power = power,
                                    alpha = alpha)
    figs = experiment.plan(plot = True)
    if show:
        save_location = 'images/plan'
        filename = '/dashboard.html'

        if not os.path.exists(save_location):
            os.mkdir(save_location)

        create_dashboard(figs, save_location + filename)
        save_images(figs, save_location)

if __name__ == '__main__':
    main()
