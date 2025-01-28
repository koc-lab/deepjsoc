# Semantic Communication over Channels with Insertions, Deletions, and Substitutions
T. Ozates and A. Koç, "Semantic Communication over Channels with Insertions, Deletions, and Substitutions," in IEEE Communications Letters, early access.

Paper link: [https://ieeexplore.ieee.org/document/10852305](https://ieeexplore.ieee.org/document/10852305)

To reproduce the results, follow these steps:

1- run `train_estimator.py` to generate the estimator file. We advise you to set `Nc` and `channel_bits`.

2- run `main.py` to train the end-to-end model. We advise to specify the relevant parameters and specify the estimator file.

3- run `performance.py` for performance evaluation.
