# Midsemester progress report

Open `main.pdf` for the current five-page report. The editable source is
`main.tex`; all three figures are in `images/`.

To rebuild with a LaTeX installation, run from this folder:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The `data/` folder contains observable estimates, fit sensitivity, the central
fit, and bootstrap standard errors used in the figures. These are preliminary
clean-model results. Raw simulation histories remain outside Git.
