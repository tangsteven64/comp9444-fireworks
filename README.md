# Environmental Sound Classification on ESC-50

COMP9444 26T3 group project, Team Fireworks.

We classify five-second environmental sound clips into the 50 classes of ESC-50 and compare five kinds of model under
one evaluation protocol: MFCC statistics with a classical classifier, a CNN trained from scratch, an ImageNet-pretrained
ResNet-18, temporal models (CRNN and BiLSTM), and an augmentation series (SpecAugment, mixup) on the from-scratch CNN.
The notebook `COMP9444-fireworks.ipynb` presents the project: code, saved outputs, results and discussion; the shared
code and the saved results it reads are in `support/`.

## Team

| Member | zID |
|---|---|
| Steven Tang | z5079275 |
| Bruce Li | z5471979 |
| Fujinmei Liu | z5618709 |
| Haole Yin | z5783825 |
| Antony Choi | z5708007 |

The method and contribution of each member are listed at the top of the notebook.

## Contents

| Path | Contents |
|---|---|
| `COMP9444-fireworks.ipynb` | the project notebook, with all outputs saved |
| `support/data.py` | shared data code: fixed split, log-mel loading, noise for the robustness test |
| `support/eval_io.py` | shared evaluation code: result format, metrics, test-fold protocol checks, results table |
| `support/notebook_setup.py` | loading of the saved results |
| `support/frozen/` | the saved results the notebook reads: the runs shown in the notebook and the small data sets for its figures and tables |
| `requirements.txt` | Python packages |

## How to run

Keep the `support/` folder next to the notebook and start Jupyter from this folder:

```bash
pip install -r requirements.txt
jupyter lab COMP9444-fireworks.ipynb      # then Kernel > Restart Kernel and Run All Cells
```

Nothing is trained when the notebook runs: every section reads the saved results in `support/frozen/`, so
Restart & Run All runs offline on a CPU in under a minute, without downloading the dataset.

Google Colab also works: open the notebook in Colab, upload the `support/` folder so that it is at `/content/support/`
(for example upload a zip of the folder and unzip it there), then Runtime > Run all.

The notebook reproduces all tables and figures from the saved results. Section 2.2 gives a preprocessing outline, and
the training and test functions of each method are in Section 4; training them needs the ESC-50 audio and the prepared
data and cache.

## Data

ESC-50 (K. J. Piczak, 2015), https://github.com/karolpiczak/ESC-50, licence CC BY-NC 3.0. The audio is not included in
this repository.
