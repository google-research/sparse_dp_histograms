"""Tools for optimize the budget split for HHHist."""

import numpy as np

import algorithms


def optimize_budget_split(**kwargs):
  """Returns the tuple (budget_split, communication) with smallest communication."""
  stage_names = ['counts', 'privtree', 'leakage']
  opt_comm = np.inf

  for leakage_budget in np.arange(0.01, 0.49, 0.01):
    split = (0.5, 0.5 - leakage_budget, leakage_budget)
    budget_split = dict(zip(stage_names, split))
    comm = algorithms.HH_Hist(
        **kwargs, budget_split=budget_split).get_expected_comm()[1]
    if comm < opt_comm:
      opt_comm = comm
      opt_budget_split = budget_split

  return opt_budget_split, opt_comm
