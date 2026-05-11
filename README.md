# STAT140 Final Project

This project analyzes whether reading different short passages changes image preferences, especially preference for penguin-related images.

## Files

- `main.py`: cleans `data.csv`, computes the main participant-level outcome, and runs the primary randomization ANOVA, pairwise tests, regression, and variance analysis.
- `individuals.py`: repeats the analysis separately for each of the 3 focal image pairs.
- `filler.py`: checks the 7 filler image pairs used to obscure the study purpose.

## Run

From this directory:

```bash
python main.py
python individuals.py
python filler.py
```

Required packages: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`, and `tqdm`. Run the following if needed.

```bash
pip install numpy pandas matplotlib statsmodels scipy tqdm
```

