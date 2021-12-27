from decimal import Decimal
from matplotlib import pyplot as plt

import argparse
import logging
import numpy as np
import os
import sys


"""
CMSC678 HW1: Numerically minimize given function
Note: Purposely not modularized main method for gradient algo
      as there is limitation to use manually calculated partial derivatives
      With manually calculated derivatives modularizing main method
      doesn't make much sense
"""


def plot_graph(xvals, yvals, xlabel, ylabel, fname):
    plt.plot(xvals, yvals, 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("{} vs {}".format(ylabel, xlabel))
    plt.savefig(fname)


def main():
    # Define function and partial derivatives
    func = lambda z1, z2: ((a - z1) ** 2) + (b * (z2 - z1 ** 2) ** 2)
    pdz1 = lambda z1, z2: (-2 * (a - z1)) - ((4 * b * z1) * (z2 - z1 ** 2))
    pdz2 = lambda z1, z2: 2 * b * (z2 - z1 ** 2)

    # Initialize values
    learning_rate = Decimal(args.learning_rate)
    expected_precision = Decimal(args.expected_precision)
    max_iterations = args.max_iterations
    a = Decimal(args.a)
    b = Decimal(args.b)
    initial_z1 = Decimal(args.initial_z_vals[0])
    initial_z2 = Decimal(args.initial_z_vals[1])

    iter_num = 1
    z1, z2 = initial_z1, initial_z2
    func_vals = [func(z1, z2)]
    while (True):
        logging.info("-" * 80)
        logging.info("Iteration: {}".format(iter_num))
        logging.info("z1 = {0:.10f}, z2 = {1:.10f}".format(z1, z2))
        logging.info("g(z1, z2) = {0:.10f}".format(func_vals[-1]))

        if (iter_num > max_iterations):
            logging.warning("Max iterations reached, stopping!")
            break

        new_z1 = z1 - learning_rate * pdz1(z1, z2)
        new_z2 = z2 - learning_rate * pdz2(z1, z2)
        func_vals.append(func(new_z1, new_z2))
        logging.info("new z1 = {0:.10f}, new z2 = {1:.10f}".format(
            new_z1, new_z2))
        logging.info("g(new_z1, new_z2) = {0:.10f}".format(func_vals[-1]))
        
        precision = abs(func_vals[iter_num] - func_vals[iter_num-1])
        if (precision <= expected_precision):
            logging.info("Expected precision achieved, stopping!")
            logging.info("Current Precision:{0:.10f}\n"
                         "Expected Precision:{0:.10f}".format(
                            precision, expected_precision))
            break

        z1, z2 = new_z1, new_z2
        iter_num += 1
    
    plot_graph(range(iter_num)[100::100], func_vals[100::100],
               "iteration", "g(z1, z2)",
               os.path.join(args.logsdir, args.graphfname))

    # Print result summary
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("*" * 80)
    logging.info("Final Values after {} iterations:-".format(iter_num))
    logging.info("z1 = {0:.10f}".format(new_z1))
    logging.info("z2 = {0:.10f}".format(new_z2))
    logging.info("g(z1, z2) = {0:.10f}".format(func_vals[-1]))
    logging.info("*" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HW1: Grdient Descent to minimize function")

    # Add script arguments
    parser.add_argument("--initial_z_vals", type=float, nargs='+',
                        required=True, help="initial values for z1, z2")
    parser.add_argument("--learning_rate", type=float, required=True,
                        help="learning rate to use")
    parser.add_argument("--max_iterations", type=int, default=1000,
                        help="Maximum iterations to run for")
    parser.add_argument("--expected_precision", type=float, default=0.0000001,
                        help="Expected precision at which loop to break")
    parser.add_argument("--a", type=int, default=1,
                        help="value to be substituted for a in given function")
    parser.add_argument("--b", type=int, default=100,
                        help="value to be substituted for b in given function")
    parser.add_argument("--logsdir", type=str, default=os.getcwd(),
                        help="Path where logs and graph need to be stored")
    parser.add_argument("--logfname", type=str, default="gradient_descent.log",
                        help="log file name")
    parser.add_argument("--graphfname", type=str, default="gradient_plot.png",
                        help="grpah file name")
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(filename=os.path.join(args.logsdir, args.logfname),
                        encoding="utf-8",
                        level=logging.INFO, filemode="w")

    main()
