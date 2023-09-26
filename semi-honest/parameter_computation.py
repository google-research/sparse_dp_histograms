"""This is a script for computing parameters for predefined epsilons, deltas.
"""
import warnings
from algorithms import Dup_Hist, nb_optimal_params

warnings.filterwarnings("ignore")

delta = 1e-11
ns = [10**i for i in range(5, 10)]
DEBUG = True

for log2_domain_size in [32, 64, 128]:
  for eps in [.5, 1., 2., 5]:
    for n in ns:
      Dup_Hist(
          eps, delta, n, log2_domain_size,
          debug=DEBUG).get_expected_comm_in_step_2()
      Dup_Hist(
          eps,
          delta,
          n,
          log2_domain_size,
          clones_optimization_active=True,
          debug=DEBUG).get_expected_comm_in_step_2()

print(f"Parameters found: {nb_optimal_params}")
