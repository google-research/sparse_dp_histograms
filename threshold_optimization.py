"""Tools for optimizing the thresholds."""

import math

from dp_accounting.common import BinarySearchParameters
from dp_accounting.common import inverse_monotone_function
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom
from scipy.stats import poisson


def additive_communication_overhead(multiplicative_overhead,
                                    T,
                                    epsilon,
                                    delta=1e-9):
  """Computes the additive overhead for SampleFrequencyDummies."""
  return (1 + multiplicative_overhead) * (0.5 * T * (T + 1)) * math.ceil(
      2 / epsilon * (1 + math.log(2 / delta)))


def num_pseudo_indices_sample_frequency_dummies(T, epsilon, delta=1e-9):
  """Computes the number of distinct pseudo indices added in SampleFrequencyDummies."""
  return T * math.ceil(2 / epsilon * (1 + math.log(2 / delta)))


def expected_num_items(num_users,
                       multiplicative_overhead,
                       T,
                       epsilon,
                       delta=1e-9):
  """Computes the expected number of items (including the input and dummy items)."""
  return ((1 + multiplicative_overhead) * num_users +
          additive_communication_overhead(
              multiplicative_overhead, T, epsilon, delta=delta))


def compute_delta_nb(r, p, T, epsilon, truncated_mass=1e-11):
  """Computes delta for negative binomial noise in the duplication setting."""
  # In scipy, the probability p is used for 1 - p in the standard definition of
  # negative binomial.
  p_scipy = 1 - p
  max_div = 0

  pruned_lower = nbinom.ppf(0.5 * truncated_mass, r * T, p_scipy)
  pruned_upper = nbinom.isf(0.5 * truncated_mass, r * T, p_scipy)
  shifted_pruned_lower = nbinom.ppf(0.5 * truncated_mass, r * (T + 1), p_scipy)
  shifted_pruned_upper = nbinom.isf(0.5 * truncated_mass, r * (T + 1), p_scipy)
  lower = int(min(pruned_lower, shifted_pruned_lower)) - 1
  upper = int(max(pruned_upper, shifted_pruned_upper)) + 1

  shifted_pruned_list = range(lower, upper)
  pruned_list = range(lower + 1, upper + 1)

  pmf = nbinom.pmf(pruned_list, r * T, p_scipy)
  shifted_pmf = nbinom.pmf(shifted_pruned_list, r * (T + 1), p_scipy)

  diff = (pmf - math.exp(epsilon) * shifted_pmf)
  diff_shifted = (shifted_pmf - math.exp(epsilon) * pmf)

  max_div = max(np.sum(diff[diff > 0]), np.sum(diff_shifted[diff_shifted > 0]))

  return max_div + truncated_mass


def compute_smallest_threshold_nb(mul_comm_overhead, epsilon, delta):
  """Computes smallest threshold for negative binomial noise in the duplication setting."""
  opt_T = math.inf
  for p in [1 - (1.1**(-a)) for a in range(1, 40)
           ] + [1.1**(-a) for a in range(1, 40)]:
    # setting this r gives the desired multiplicative communication overhead
    r = (1 - p) / p * mul_comm_overhead
    search_param = BinarySearchParameters(1, opt_T, 50, discrete=True)
    T = inverse_monotone_function(
        lambda x: compute_delta_nb(
            r, p, x, epsilon, truncated_mass=0.01 * delta), delta, search_param)
    if T is not None and T < opt_T:
      opt_T = T
      opt_r = r
      opt_p = p
  return opt_T, opt_r, opt_p


def optimize_negative_binomial_parameter(num_users,
                                         epsilon,
                                         delta=1e-9):
  """Returns optimal (r, p, T, expected_num_items) for given number of users.

  If not found, return None.
  """

  def expected_num_items_given_mul_comm_overhead(mul_comm_overhead):
    T, r, p = compute_smallest_threshold_nb(
        mul_comm_overhead, epsilon, delta=delta)
    return expected_num_items(
        num_users, mul_comm_overhead, T, epsilon, delta=delta)

  res = minimize_scalar(
      expected_num_items_given_mul_comm_overhead,
      bounds=(0.0001, 1),
      method='bounded',
      options={'xatol': 1e-2})
  mul_comm_overhead = res.x
  expected_num_items_ = res.fun
  T, r, p = compute_smallest_threshold_nb(
      mul_comm_overhead, epsilon, delta=delta)
  return (r, p, T, expected_num_items_,
          num_pseudo_indices_sample_frequency_dummies(T, epsilon, delta))


def prob_bad(epsilon, delta, expected_clones, q):
  """Computes the probability of high privacy loss.

  Computes the probability that privacy loss is more than epsilon for a given
  intensity of clones.

  Args:
    epsilon: DP parameter.
    delta: DP parameter (used to decide required accuracy).
    expected_clones: The intensity of the number of clones of each type.

  Returns:
    Probability that the privacy loss exceeds epsilon.
  """
  if expected_clones < 0:
    return 1
  rv = poisson(expected_clones)
  upper_bound = rv.isf(delta / 4)
  if np.isinf(upper_bound):
    return 1
  total_prob = 0
  prob = rv.pmf(list(range(math.ceil(upper_bound))))
  tsf = np.cumsum(prob[::-1])[::-1]
  exp_epsilon = math.exp(epsilon)
  for c in range(math.ceil(upper_bound)):
    weighted_c = (1 - q) * c / q
    a = exp_epsilon * (weighted_c) - (weighted_c + 1)
    prob_c = 0
    for b in range(math.ceil(upper_bound)):
      if math.floor(a) < len(tsf):
        prob_c += prob[b] * tsf[math.floor(a)]
      a += exp_epsilon
    total_prob += prob[c] * prob_c
  return total_prob


clone_cache = {}


def required_clones(epsilon, delta, q):
  """Numerically bounds the number of clones requried,

  This code is based on the Poissonized version of the algorithm. It also
  assumes the same number of clones of each type will be wanted, and doesn't
  take advantage of small q. I think these losses will be small.

  Args:
    epsilon: DP parameter for hiding multiplicities.
    delta: DP parameter for hiding multiplicities.

  Returns:
    The smallest poisson intensity that gives the DP guarantee for all q.
  """
  if q < delta:
    return 0
  if (epsilon, delta) not in clone_cache:
    clone_cache[(epsilon, delta)] = {}
  current_cache = clone_cache[(epsilon, delta)]
  # For caching purposes, only allow q to be a power of 1.1
  q = (1.1)**math.ceil(math.log(q, 1.1))
  if q not in current_cache:
    l = 0
    u = math.inf
    for oldq in current_cache:
      if oldq < q:
        l = max(l, current_cache[oldq])
      else:
        u = min(u, current_cache[oldq])
    if u > 0.002:
      if u == math.inf:
        u = l + 1
        while prob_bad(epsilon, delta * 0.01, u, q) > delta * 0.99:
          u *= 2
          u -= l
      elif l == 0:
        l = u - 1
        while prob_bad(epsilon, delta * 0.01, l, q) < delta * 0.99:
          l *= 2
          l -= u
        l = max(l, 0)
      while u - l > 0.001:
        m = (u + l) / 2
        if prob_bad(epsilon, delta * 0.01, m, q) > delta * 0.99:
          l = m
        else:
          u = m
    current_cache[q] = u
  return current_cache[q]


def optimize_Tpp(epsilon,
                 delta,
                 r,
                 p,
                 Tp,
                 num_users):
  """Output the T'' that minimizes the communication cost when fixing other parameters."""
  # In scipy, the probability p is used for 1 - p in the standard definition
  # of negative binomial.
  p_scipy = 1 - p
  old_dummies_per_entry = math.ceil(2 / epsilon * math.log(1 / delta))
  blowup = 1 + nbinom.mean(r, p_scipy)

  total_upper_bound = Tp + math.ceil(nbinom.isf(delta * 0.01, r * Tp, p_scipy))

  eta = [0] * total_upper_bound
  # This is the expected number of messages without using clones.
  best_num_msg = blowup * (
      num_users + 0.5 * Tp * (Tp + 1) * old_dummies_per_entry)
  optimal_Tpp = Tp

  # This is a hacky if to avoid getting into the following code which requires
  # a running time of O(T'^2).
  if Tp > 10_000:
    return best_num_msg, optimal_Tpp

  fi = nbinom.pmf(
      list(range(-Tp, total_upper_bound - Tp)), r * Tp, p_scipy)
  for i in range(Tp - 1, 0, -1):
    # Compute the communication for Tpp = i.

    # Update etas.
    fip = fi
    fi = nbinom.pmf(list(range(-i, total_upper_bound - i)), r * i, p_scipy)
    gamma_i = np.minimum(fi, fip)
    alpha_i = fi - gamma_i
    beta_i = fip - gamma_i
    q_i = sum(alpha_i)
    gamma_i /= (1 - q_i)
    alpha_i /= q_i
    beta_i /= q_i

    clones_i = required_clones(epsilon, delta * 0.99, q_i)
    eta = np.maximum(eta, clones_i * (gamma_i + alpha_i + beta_i))

    # Expected number of messages if we set Tpp = i
    curr_num_msg = blowup * (num_users + 0.5 * i *
                             (i + 1) * old_dummies_per_entry) + sum(
                                 i * eta[i] for i in range(len(eta)))

    if curr_num_msg < best_num_msg:
      best_num_msg = curr_num_msg
      optimal_Tpp = i
      # The number of distinct pseudo-indices added by Figure 8 and Figure 10.
      num_pseudo_indices = sum(eta) + i * old_dummies_per_entry

  return best_num_msg, num_pseudo_indices, optimal_Tpp


def optimize_negative_binomial_parameter_with_clones(num_users,
                                                     epsilon,
                                                     delta=1e-9):
  """Returns optimal (r, p, T', T'', expected_num_items) for given number of users.

  If not found, return None.
  """

  def expected_num_items_given_mul_comm_overhead(mul_comm_overhead):
    T, r, p = compute_smallest_threshold_nb(
        mul_comm_overhead, epsilon, delta=delta)
    return optimize_Tpp(epsilon, delta, r, p, T, num_users)[0]

  res = minimize_scalar(
      expected_num_items_given_mul_comm_overhead,
      bounds=(0.0001, 1),
      method='bounded',
      options={'xatol': 1e-2})
  mul_comm_overhead = res.x
  T, r, p = compute_smallest_threshold_nb(
      mul_comm_overhead, epsilon, delta=delta)
  expected_num_items_, num_pseudo_indices_, Tpp = optimize_Tpp(
      epsilon, delta, r, p, T, num_users)
  return (r, p, T, Tpp, expected_num_items_, num_pseudo_indices_)
