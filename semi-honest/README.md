# Experiments in *Distributed, Private, Sparse Histograms in the Two-Server Model*.

## Code Overview

Our code is organized as follows:

*   `algorithms.py` cotains the main classes for the algorithms, including our
    main protocol presented in Figure 6 (`Dup_Hist`) and the Heavy-Hitter-based
    protocol described in Appendix E (`HH_Hist`).

*   `threshold_optimization.py` contains helper functions for computing and
    optimizing the noise parameters for our protocol.

*   `parameter_computation.py` is a script for pre-computing the noise
    parameters, which is a computationally

*   extensive operation. The parameters computed from this script are recorded
    in the `nb_optimal_params` variable in `algorithms.py`. -
    `experiments.ipynb` is a colab that computes the communication of each
    protocol for several parameter settings and generate the plots presented in
    our paper.

## Instructions

To install dependencies, run `pip install -r requirements.txt` and the colab
notebook for producing the plots can be then run using the command `jupyter
notebook experiments.ipynb`.
