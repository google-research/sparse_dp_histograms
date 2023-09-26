"""Algorithm classes that allow us to compute communication complexity."""

import math
import numpy as np
import threshold_optimization

# Precomputed by running parameter_computation.py.
nb_optimal_params = {
    (100000, 0.125, 1.887703343990727e-12, False):
        (0.030105313128493936, 0.9705916506294485, 1031, 488068065.4618946,
         474260),
    (1000000, 0.125, 1.887703343990727e-12, False):
        (0.030105209135487632, 0.9705916506294485, 1031, 489861456.8989779,
         474260),
    (10000000, 0.125, 1.887703343990727e-12, False):
        (0.03010416920542462, 0.9705916506294485, 1031, 507795031.4840071,
         474260),
    (100000000, 0.125, 1.887703343990727e-12, False):
        (0.03019476836733794, 0.9705916506294485, 1029, 686353018.4123209,
         473340),
    (1000000000, 0.125, 1.887703343990727e-12, False):
        (0.026493994903289377, 0.9644158972616327, 1300, 2386372265.094337,
         598000),
    (100000, 0.25, 1.3447071068499755e-12, False):
        (0.06684993238906588, 0.9369605913687152, 533, 66303964.08774587, 124189
        ),
    (1000000, 0.25, 1.3447071068499755e-12, False):
        (0.06684819830042446, 0.9369605913687152, 533, 68097320.77445571, 124189
        ),
    (10000000, 0.25, 1.3447071068499755e-12, False):
        (0.06705512756241701, 0.9369605913687152, 532, 85924236.65859117, 123956
        ),
    (100000000, 0.25, 1.3447071068499755e-12, False):
        (0.058117041589058484, 0.9306566505055868, 631, 260695804.53277218,
         147023),
    (1000000000, 0.25, 1.3447071068499755e-12, False):
        (0.034223093003670484, 0.907704001822936, 1225, 1570428437.6025038,
         285425),
    (100000, 0.5, 5.960146101105878e-13, False):
        (0.13916181171000205, 0.8771540264263278, 276, 9344493.64557762, 33120),
    (1000000, 0.5, 5.960146101105878e-13, False):
        (0.1391348765175906, 0.8771540264263278, 276, 11137707.341432473, 33120
        ),
    (10000000, 0.5, 5.960146101105878e-13, False):
        (0.1398574835133815, 0.8648694290689605, 296, 28947470.91555249, 35520),
    (100000000, 0.5, 5.960146101105878e-13, False):
        (0.0912294579682985, 0.802155331099865, 580, 164686186.4770844, 69600),
    (1000000000, 0.5, 5.960146101105878e-13, False):
        (0.053722780224405756, 0.7513148009015775, 1192, 1261476194.8691647,
         143040),
    (100000, 1.25, 3.3464254621424277e-14, False):
        (0.6166523791698326, 0.6144567105704686, 118, 936098.6123818121, 6254),
    (1000000, 1.25, 3.3464254621424277e-14, False):
        (0.5894383692967643, 0.5644739300537771, 139, 2673609.9077706253, 7367),
    (10000000, 1.25, 3.3464254621424277e-14, False):
        (0.7156277240787217, 0.3186308177103565, 259, 15728210.97333553, 13727),
    (100000000, 1.25, 3.3464254621424277e-14, False):
        (0.4638256300454094, 0.23939204936916336, 532, 123209532.15263723, 28196
        ),
    (1000000000, 1.25, 3.3464254621424277e-14, False):
        (0.8786608067685446, 0.06934334949441323, 1125, 1101235732.455152, 59625
        ),
    (100000, 0.125, 1.887703343990727e-12, True):
        (0.03012936674566541, 0.9705916506294485, 1030, 10, 801332.2387550097,
         10161.42304179463),
    (1000000, 0.125, 1.887703343990727e-12, True):
        (0.027093466457007557, 0.9644158972616327, 1279, 12, 2511797.562664834,
         11721.871075200128),
    (10000000, 0.125, 1.887703343990727e-12, True):
        (0.015339638405803472, 0.9526375592552331, 2567, 27, 15286582.237137718,
         21262.680725303813),
    (100000000, 0.125, 1.887703343990727e-12, True):
        (0.008212614277128221, 0.9426914466988321, 5402, 57, 121612638.57896145,
         41807.422527743634),
    (1000000000, 0.125, 1.887703343990727e-12, True):
        (0.004484861464112565, 0.9426914466988321, 9583, 105,
         1097689179.1464372, 76080.32692924698),
    (100000, 0.25, 1.3447071068499755e-12, True):
        (0.05918505121428825, 0.9426914466988321, 540, 4, 459733.9808169567,
         3537.0911752208526),
    (1000000, 0.25, 1.3447071068499755e-12, True):
        (0.044502406924093435, 0.9237223155561455, 834, 7, 1991508.6531552402,
         4769.441918855013),
    (10000000, 0.25, 1.3447071068499755e-12, True):
        (0.025003587361154718, 0.8984744020052295, 1765, 17, 13712828.237244569,
         8999.699808110583),
    (100000000, 0.25, 1.3447071068499755e-12, True):
        (0.014197459005845755, 0.8771540264263278, 3626, 36, 115744791.709561,
         17312.395655553173),
    (1000000000, 0.25, 1.3447071068499755e-12, True):
        (0.006212558737322035, 0.8771540264263278, 8033, 81, 1069629490.3788761,
         38911.15656123108),
    (100000, 0.5, 5.960146101105878e-13, True):
        (0.12925663281800814, 0.8771540264263278, 290, 2, 324615.15706457524,
         1410.6981957350679),
    (1000000, 0.5, 5.960146101105878e-13, True):
        (0.09612178509311013, 0.802155331099865, 556, 4, 1717138.2351717537,
         2352.939646665407),
    (10000000, 0.5, 5.960146101105878e-13, True):
        (0.054149054059196206, 0.7606079506308366, 1131, 9, 12794538.583387125,
         4506.284856883392),
    (100000000, 0.5, 5.960146101105878e-13, True):
        (0.02778781083433352, 0.7366687456939203, 2377, 18, 112001125.71075003,
         9252.225113088438),
    (1000000000, 0.5, 5.960146101105878e-13, True):
        (0.014084253552601384, 0.7103356202633122, 5214, 44, 1054117246.1419548,
         19872.66619507891),
    (100000, 1.25, 3.3464254621424277e-14, True):
        (0.6107412218862798, 0.533492619790267, 148, 1, 237112.1904102585,
         824.5241678596877),
    (1000000, 1.25, 3.3464254621424277e-14, True):
        (0.5933465981048764, 0.3169865446349295, 306, 2, 1476664.0152209336,
         1591.5152303012856),
    (10000000, 1.25, 3.3464254621424277e-14, True):
        (0.6332061977925737, 0.16350799082655781, 619, 4, 11951009.9877106,
         3160.7401428079797),
    (100000000, 1.25, 3.3464254621424277e-14, True):
        (0.4440291842902595, 0.11167815779424749, 1311, 10, 108509565.14580551,
         6534.27958947327),
    (1000000000, 1.25, 3.3464254621424277e-14, True):
        (0.40658713465739443, 0.05730855330116795, 2896, 20, 1038579854.1154442,
         14411.804211487708)
}

# Timings for different operations, in seconds, and sizes for different ciphertexts, in bytes.
microbenchmarks = {
    'ElGamalSize': 64,
    'ElGamalEncrypt': 102e-6,
    'ElGamalDecrypt': 50.6e-6,
    'ElGamalRerandomize': 512e-9,
    'ElGamalExp': 101e-6,
    'PaillierSize': 256,
    'PaillierEncrypt': 919e-6,
    'PaillierDecrypt': 368e-6,
    'PaillierAdd': 2.3e-6,
    'PaillierRerandomize': 2.63e-6,
    'ExponentialElGamalSize': 64,
    'ExponentialElGamalEncrypt': 152e-6,
    'ExponentialElGamalDecrypt': 101e-6,
    'ExponentialElGamalAdd': 537e-9,
    'ExponentialElGamalRerandomize': 512e-9,
    'HybridEncrypt':
        102e-6,  # Using just ElGamal here, since ciphertexts are small and symmetric crypto is cheap.
    'HybridDecrypt': 50.6e-6,
}


def substitution_to_add_remove(eps_sub, delta_sub):
  """Returns eps, delta for which (eps, delta)-add/remove DP implies (eps_sub, delta_sub)-substitution DP."""
  # Use formula from footnote 1 in the paper.
  eps = eps_sub / 2
  delta = delta_sub / (1 + math.exp(eps_sub))
  return eps, delta


class Hist:
  """
   Class representing a run of a protocol for building a histogram.
  """

  def __init__(
      self,
      epsilon,
      delta,
      n,
      log2_domain_size,
      T,
      he_mode='ExponentialElGamal',  # Possible values: "ExponentialElGamal", "Paillier"
      debug=False
  ):  # one domain element containing h(m)^k, and one Elgamal ciphertext (encrypting m with shared key)
    self.epsilon, self.delta = substitution_to_add_remove(epsilon, delta)
    self.n = n
    self.log2_domain_size = log2_domain_size
    self.T = T
    self.bitwith_for_counts = 8 * np.ceil(np.log2(self.n) / 8.)
    self.elgamal_ciphertext = microbenchmarks['ElGamalSize'] * 8
    self.debug = debug
    self.expected_dummy_pseudo_indices = None
    self.additive_he_ciphertext_size_in_bits = microbenchmarks[he_mode +
                                                               'Size'] * 8
    self.encrypted_hash_size_in_bits = 2 * self.elgamal_ciphertext
    self.he_mode = he_mode

  def get_expected_comm_in_step_1(self):
    """
       Clients -> A
       ------------
       Clients send contributions.
       In the HH based protocol they also send DPF keys.
    """
    # In the first round the clients send their encrypted messages.
    return self.n * (self.encrypted_hash_size_in_bits + self.additive_he_ciphertext_size_in_bits)

  def get_expected_comm_in_step_2(self):
    """
       A -> B
       ------
       A sends to B encrypted input & dummies,
       possibly after running a HH subprotocol.
       A gets an anonymous histogram of indices
       and HE-encrypted values.
    """
    # Protocol specific. Subclass should implement this.
    raise NotImplementedError

  def get_expected_comm_in_step_3(self):
    """
      B -> A
      ------
      B sends noisy encrypted counts to A,
      including some fake counts from SampleBuckets.

      Additionally, B sends elGamal encryptions of all different indices,
      for A to select the ones that need to be decrypted.

      Also note that we're assuming a worst-case input with n different
      elements. For specific input distributions (i.e. those with few HHs),
      these can possibly be optimized further; such optimization is not included
      in the current code.
    """
    epsilon_leakage = self.budget_split['leakage'] * self.epsilon
    delta_leakage = self.budget_split['leakage'] * self.delta
    # Number of expected dummies added by SampleBuckets (Step 2 in Figure 12)
    # assuming Delta = 1.
    num_dummies_for_thresholding = math.ceil(1 + np.log(1 / delta_leakage) /
                                             epsilon_leakage)
    self.expected_num_dummies_for_thresholding_in_step_3 = num_dummies_for_thresholding
    return (num_dummies_for_thresholding + self.n +
            self.expected_dummy_pseudo_indices) * \
                (self.additive_he_ciphertext_size_in_bits + self.elgamal_ciphertext)

  def get_expected_comm_in_step_4(self):
    """
      A -> B
      ------
      A sends to B ElGamal encryptions of all indices whose value
      is above the threshold tau.
      Note that all dummies are discarded by the thresholding step.
    """
    return self.n / self.tau * self.elgamal_ciphertext

  def get_expected_comm_in_step_5(self):
    """
      B -> A
      ------
      B replies with partial decryptions for the ElGamal encryptions
      received from A in the previous step.
    """
    return self.get_expected_comm_in_step_4()

  def get_expected_comm(self, unit='bits', parties={'Helper 1', 'Helper 2'}):
    """
      Returns an upper bound on expected comm., for the helpers.
      That is, the total amount of information going from A to B, or viceversa.
      Valid values in `parties` are 'Helper 1', 'Helper 2, and 'Client'.
      Defaults to only the two helpers, since that is what we want to optimize
      parameters for.
    """
    if unit == 'KB':
      div = (8 * 10**3)
    if unit == 'bits':
      div = 1
    if unit == 'bytes':
      div = 8
    comm_per_step = dict()
    for i in range(1, 6):
      comm_per_step[i] = 0
    if 'Client' in parties:
      comm_per_step[1] = self.get_expected_comm_in_step_1()
    if 'Helper 1' in parties:
      comm_per_step[2] = self.get_expected_comm_in_step_2()
    if 'Helper 2' in parties:
      comm_per_step[3] = self.get_expected_comm_in_step_3()
    if 'Helper 1' in parties:
      comm_per_step[4] = self.get_expected_comm_in_step_4()
    if 'Helper 2' in parties:
      comm_per_step[5] = self.get_expected_comm_in_step_5()
    for i in range(1, 6):
      comm_per_step[i] /= div
    total_comm = sum(comm_per_step[i] for i in range(1, 6))
    return comm_per_step, total_comm

  def get_expected_computation(self, parties):
    """
      Returns the number of seconds of computation needed by the `parties`, split into
      offline and online time.
    """
    # Implemented by subclass
    raise NotImplementedError


class Dup_Hist(Hist):
  """
    Class representing a run of the protocol based on duplicating hashed client
    contributions. The protocol is presented in Figure 6 of our paper.
  """

  def __init__(
      self,
      epsilon,
      delta,
      n,
      log2_domain_size,
      clones_optimization_active=False,
      he_mode='ExponentialElGamal',  # Possible values: "ExponentialElGamal", "Paillier"
      debug=False):
    self.budget_split = {
        'leakage': 0.5,
        'counts': 0.5,
    }
    assert (.9999 < sum(self.budget_split.values()) <= 1)
    super(Dup_Hist, self).__init__(
        epsilon,
        delta,
        n,
        log2_domain_size,
        T=None,
        he_mode=he_mode,
        debug=debug)
    delta_counts = self.budget_split['counts'] * self.delta
    epsilon_counts = self.budget_split['counts'] * self.epsilon
    self.tau = 2 * np.log(1. / delta_counts) / epsilon_counts
    self.with_clones = clones_optimization_active
    self.expected_num_dummies_step_2 = None

  def get_expected_comm_in_step_2(self, p=None, T=None):
    n = self.n
    epsilon = self.budget_split['leakage'] * self.epsilon
    delta = self.budget_split['leakage'] * self.delta
    if not ((n, epsilon, delta, self.with_clones) in nb_optimal_params):
      print(f'Finding optimal parameters for n = {n}, epsilon = {epsilon}, '
            f'delta = {delta}, with_clones = {self.with_clones}')
      if self.with_clones:
        nb_optimal_params[(n, epsilon, delta, self.with_clones)] = (
            threshold_optimization
            .optimize_negative_binomial_parameter_with_clones(
                n, epsilon, delta))
      else:
        nb_optimal_params[(n, epsilon, delta, self.with_clones)] = (
            threshold_optimization.optimize_negative_binomial_parameter(
                n, epsilon, delta))
    if self.with_clones:
      r, p, self.Tpp, self.T, expected_num_items, self.expected_dummy_pseudo_indices = nb_optimal_params[
          (n, epsilon, delta, self.with_clones)]
    else:
      r, p, self.T, expected_num_items, self.expected_dummy_pseudo_indices = nb_optimal_params[
          (n, epsilon, delta, self.with_clones)]
      self.Tpp = None
    self.expected_num_dummies_step_2 = np.ceil(expected_num_items) - self.n

    if self.debug:
      print('Dup-Hist:')
      mult_overhead = r * p / (1 - p)
      print('\tT = {}, mult_overhead = {:.2f}'.format(self.T, mult_overhead))
      if self.with_clones:
        print('\tTpp = {}'.format(self.Tpp))
      print('\tExpected number of dummies = {}'.format(
          self.expected_num_dummies_step_2))
    return np.ceil(expected_num_items) * (
        self.additive_he_ciphertext_size_in_bits +
        self.encrypted_hash_size_in_bits)

  def get_expected_computation(self, parties=set()):
    # Some commonly used measures.
    time_per_client = 2 * microbenchmarks['ElGamalEncrypt'] + \
      microbenchmarks[self.he_mode + 'Encrypt'] + microbenchmarks['HybridEncrypt']

    offline_time, online_time, bytes_sent = 0, 0, 0

    # Step 1. The time is the total time for all clients.
    if 'Client' in parties:
      online_time += self.n * time_per_client

    self.get_expected_comm_in_step_2()
    num_dummies_step_2 = self.expected_num_dummies_step_2
    num_different_indices_step_2 = self.n + self.expected_dummy_pseudo_indices

    self.get_expected_comm_in_step_3()
    num_dummies_step_3 = self.expected_num_dummies_for_thresholding_in_step_3

    # Step 2.
    num_ciphertexts = self.n + num_dummies_step_2
    offline_time_per_rerandomization = time_per_client
    online_time_per_randomization = (
        2 * microbenchmarks['ElGamalRerandomize'] +
        microbenchmarks[self.he_mode + 'Rerandomize'])

    if 'Helper 1' in parties:
      # Exponentiate the first component of all ciphertexts
      online_time += num_ciphertexts * microbenchmarks['ElGamalExp']

      # Rerandomize all ciphertexts. We use time_per_client for the offline time,
      # which corresponds encrypting zeros in the case of rerandomization.
      offline_time += num_ciphertexts * offline_time_per_rerandomization
      online_time += num_ciphertexts * online_time_per_randomization

    # Step 3.
    if 'Helper 2' in parties:
      # Decrypt first and third components.
      online_time += num_ciphertexts * (microbenchmarks['ElGamalDecrypt'] +
                                        microbenchmarks['HybridDecrypt'])

      # Homomorphically add up ciphertexts in the same buckets. Worst case: all in
      # a single bucket -> num_ciphertexts additions.
      online_time += num_ciphertexts * microbenchmarks[self.he_mode + 'Add']

    # After aggregating, all dummies that are the same will disappear, so the
    # number of ciphertexts we have is only self.n + number of different
    # dummies.
    num_ciphertexts = self.n + self.expected_dummy_pseudo_indices

    # Step 3d: Dummies that Server 2 adds
    num_ciphertexts += num_dummies_step_3
    if 'Helper 2' in parties:
      # Add dummies for low counts. Note that these are only HE ciphertexts.
      offline_time += num_dummies_step_3 * microbenchmarks[self.he_mode +
                                                           'Encrypt']

    # Step 4.
    # Thresholding Step 1.
    if 'Helper 2' in parties:
      # Add additive noise using HE. This is done by first encrypting the noise
      # value and then homomorphically adding it. TODO: Is there anything faster?
      # Worst case: Number of buckets = num_ciphertexts
      offline_time += num_ciphertexts * microbenchmarks[self.he_mode +
                                                        'Encrypt']
      online_time += num_ciphertexts * microbenchmarks[self.he_mode + 'Add']

      # Rerandomize elgamal ciphertexts corresponding to indices,
      # and encrypt dummies added in this step, using a fake index.
      offline_time += (num_ciphertexts) * microbenchmarks['ElGamalEncrypt']
      online_time +=  (num_ciphertexts - num_dummies_step_3) * microbenchmarks['ElGamalRerandomize']

      # Rerandomize and send all ciphertexts corresponding to values.
      online_time += num_ciphertexts * microbenchmarks[self.he_mode +
                                                       'Rerandomize']

    # Thresholding Step 2.
    if 'Helper 1' in parties:
      # Decrypt values
      online_time += num_ciphertexts * microbenchmarks[self.he_mode + 'Decrypt']

    # After thresholding, everything below tau disappears. In the worst case, we
    # have all clients split evenly across buckets of size exactly tau.
    num_ciphertexts = self.n / self.tau

    # Step 5.
    if 'Helper 1' in parties:
      # Rerandomize elgamal ciphertexts corresponding to the final indices / bucket IDs.
      offline_time += num_ciphertexts * microbenchmarks['ElGamalEncrypt']
      online_time +=  num_ciphertexts * microbenchmarks['ElGamalRerandomize']

    # Step 6.
    if 'Helper 2' in parties:
      # partial decryption of final indices / bucket IDs.
      online_time += num_ciphertexts * microbenchmarks['ElGamalDecrypt']

    # Step 7.
    if 'Helper 1' in parties:
      # Decrypt final indices / bucket IDs.
      online_time += num_ciphertexts * microbenchmarks['ElGamalDecrypt']

    return (offline_time, online_time)


class HH_Hist(Hist):
  """
    Class representing a run of the protocol based on finding T-heavy hitters
    using DPFs. This protocol is described in Appendix E.
  """

  def __init__(
      self,
      epsilon,
      delta,
      n,
      log2_domain_size,
      h=None,  # h is the number of heavy hitters, at most n/T
      log2_max_beta=6,
      debug=False,
      budget_split=None):
    super(HH_Hist, self).__init__(
        epsilon, delta, n, log2_domain_size, T=None, debug=debug)

    self.budget_split = (
        budget_split or {'counts': 0.5, 'privtree': 0.3, 'leakage': 0.2})
    assert (.9999 < sum(self.budget_split.values()) <= 1)
    delta_counts = self.budget_split['counts'] * self.delta
    epsilon_counts = self.budget_split['counts'] * self.epsilon
    self.tau = np.ceil(2 * np.log(1. / delta_counts) / epsilon_counts)

    class PrivTreeParams:
      """Parameters for the PrivTree algorithm.

      See the following paper for more detail:
      Jun Zhang, Xiaokui Xiao, and Xing Xie. 2016. PrivTree: A Differentially
      Private Algorithm for Hierarchical Decompositions. In Proceedings of the
      2016 International Conference on Management of Data (SIGMOD ’16). 155–170.
      """

      @classmethod
      def get_default_params(cls, beta=2):
        gam = np.ceil(np.log(beta))
        epsilon_privtree = self.budget_split['privtree'] * self.epsilon
        lamb = (2 * np.exp(gam) - 1) / (np.exp(gam) - 1) * 1. / epsilon_privtree
        b = lamb * gam
        theta = 0
        return b, lamb, epsilon, gam, theta

      def __init__(
          self,
          b,
          lamb,
          epsilon,
          gam,
          theta,
          beta=2
      ):  # lamb, epsilon, gamma are lambda, epsilon, gamma in the PrivTree paper
        self.b = b  # bias factor, \delta in the PrivTree paper
        self.lamb = lamb  # Scale of Laplace
        self.epsilon = epsilon
        self.gam = gam  # factor relating lambda and b as b = gam * lamb
        self.beta = beta  # branching factor in the tree
        assert gam > 0, gam
        assert lamb >= (2 * np.exp(gam) - 1) / (np.exp(gam) - 1) * 1 / epsilon
        assert b >= gam * lamb, '{} != {}'.format(b, gam * lamb)
        self.theta = theta  # Internal PrivTree thresholdself.b * log_domain_size

      def get_T_lower_bound(self, delta, d):
        # Returns value of T so that a T-heavy hitter is missed w.p. < delta
        T = np.ceil(lamb * (np.log(1 / delta) + np.log(2)) + self.b * (d - 1) +
                    self.theta)
        return T

    # Find value of beta that minimizes communication.
    self.log2_max_beta = log2_max_beta
    self.log2_domain_size = log2_domain_size
    best_beta = best_comm = None
    for beta in [2**i for i in range(1, self.log2_max_beta + 1)]:
      b, lamb, epsilon, gam, theta = PrivTreeParams.get_default_params(beta)
      params = PrivTreeParams(b, lamb, epsilon, gam, theta, beta)
      # d is the number of levels in the PrivTree tree
      d = log2_domain_size / np.log2(beta)
      T = params.get_T_lower_bound(self.delta, d)
      step_2_comm = self.get_expected_comm_in_step_2(
          h if h else np.ceil(self.n / T), T, d)
      if not best_beta or best_comm > step_2_comm:
        best_beta, best_comm = beta, step_2_comm
    self.beta = best_beta
    self.d = log2_domain_size / np.log2(self.beta)
    b, lamb, epsilon, gam, theta = PrivTreeParams.get_default_params(self.beta)
    params = PrivTreeParams(b, lamb, epsilon, gam, theta, self.beta)
    best_T = params.get_T_lower_bound(self.delta, self.d)
    self.T = best_T
    # h is a bound on the number of indices with multiplicity >= T
    self.h = min(h if h else 0, np.ceil(n / self.T))
    if self.debug:
      print('HH-Hist:')
      print('\tbeta = {}, T = {}, h = {}'.format(self.beta, self.T, self.h))

  def get_num_dummies_to_mask_low_counts(self, T=None):
    if not T:
      T = self.T
    epsilon_low_counts = self.budget_split['leakage'] * self.epsilon
    delta_low_counts = self.budget_split['leakage'] * self.delta
    num_dummies_to_mask_low_counts = (0.5 * T * (T + 1)) * \
      np.ceil(2 / epsilon_low_counts * np.log(1./delta_low_counts)) # Expectation of a shifted Laplace
    self.expected_dummy_pseudo_indices = (
        threshold_optimization.num_pseudo_indices_sample_frequency_dummies(
            T, epsilon_low_counts, delta_low_counts))
    return num_dummies_to_mask_low_counts

  def get_expected_comm(self, unit='bits'):
    if unit == 'KB':
      div = (8 * 10**3)
    if unit == 'bits':
      div = 1
    if unit == 'bytes':
      div = 8
    if self.debug:
      print('\tprotocol communication in {}, per client (HH_Hist) = {:.2f}'
            .format(unit,
                    self.get_privtree_comm() / self.n / div))
    return super(HH_Hist, self).get_expected_comm(unit)

  def get_num_dummies_to_mask_high_counts(self, h=None):
    # This is an optimistic estimate that does not account for false positives
    # from PrivTree algorithm, which can contribute to additional communication.
    if not h:
      h = self.h
    epsilon_high_counts = self.budget_split['leakage'] * self.epsilon
    delta_high_counts = self.budget_split['leakage'] * self.delta
    return np.ceil(
        h * np.ceil(np.log(1. / delta_high_counts) / epsilon_high_counts))

  def get_privtree_comm(self, h=None, d=None):
    if not h:
      h = self.h
    if not d:
      d = self.d
    # maximum number of opened nodes in the hierarchical PrivTree-based HH
    # (this is a rough upper bound, TODO: sharpen it)
    hh_max_opened_nodes = 2 * h * d

    # This counts and gates in a GC that
    #   (a) reconstructs bit-wise shares from DPF outputs,
    #   (b) compares, and
    #   (c) thresholds
    # Log(n) here is optimistic, as we have not accounted for the noise.
    # However, it is likely that the overhead due to noise is (multiplicatively)
    # small.
    num_and_gates_for_privtree = 3 * np.ceil(np.log2(self.n))
    garbled_gate_size_in_bits = 256
    comparison_GC_in_bits = num_and_gates_for_privtree * 256
    share_size_in_bits = 64

    return hh_max_opened_nodes * (share_size_in_bits + comparison_GC_in_bits)

  def get_expected_comm_in_step_2(self, h=None, T=None, d=None):
    privtree_comm_in_bits = self.get_privtree_comm(h, d)
    return privtree_comm_in_bits + \
          (self.n + self.get_num_dummies_to_mask_low_counts(T) +
              self.get_num_dummies_to_mask_high_counts(h)) * \
              (self.additive_he_ciphertext_size_in_bits +
               self.encrypted_hash_size_in_bits)
