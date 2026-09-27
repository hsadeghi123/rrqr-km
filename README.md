# RRQR-KM

Companion code for the paper:

H. Sadeghi, Volume bounds for strong rank-revealing QR of arbitrary shape,
and a deterministic K-means initialization.

Author: Hossein Sadeghi (hsadeghi@iasbs.ac.ir)
ORCID: https://orcid.org/0000-0003-1903-1627

https://doi.org/10.5281/zenodo.22996265
GitHub: https:github.com/hsadeghi123/rrqr-km

## Contents
   'rrqr_km.py'- the core algorithm 
 
- 'run_experiments.py' — UCI benchmarks and tables

   'exp_kahan_f.py` — UCI benchmarks and tables

    kahan_f_results
    summary
    tables 
     .....
## How to run

Python 3.10 or newer.

    pip install -r requirements.txt

    python run_experiments.py

    python exp_kahan_f.py
    


Randomized baselines: 20 restarts.
Default RRQR parameter: f = sqrt(n).

## Data

Nine UCI data sets (Banknote, Glass, Ionosphere, Iris, Seeds, WDBC,
Wholesale, Wine, Yeast). Download from
https://archive.ics.uci.edu
Place them where the script expects (see comments in the code).
The scripts apply per-feature standardization internally; do not
standardize twice.
