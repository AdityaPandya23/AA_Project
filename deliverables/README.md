# Deliverables: final report and presentation

| File | What |
|---|---|
| `Project_Report.pdf` | Final project report (16 pages), compiled from `report/main.tex` |
| `presentation/Imagined_Shape_EEG_Presentation.pptx` | 16-slide presentation for the evaluation, with speaker notes on every slide |
| `report/main.tex`, `report/figures/` | LaTeX source and figures (figures come from `results/final/figures/`; the logo comes from `Advance_Algo_Project.pdf`) |

## Rebuilding the PDF

Any LaTeX engine works; the source uses only standard packages.

- pdfLaTeX: `cd report && pdflatex main.tex && pdflatex main.tex`
- Tectonic (used for this PDF): `cd report && tectonic main.tex`
- Overleaf: upload the `report/` folder and compile.

`hyperref` is left out because the Tectonic 0.15 build used here fails on it. With
pdfLaTeX or Overleaf you can add `\usepackage[hidelinks]{hyperref}` back.

Every number in both files comes from `docs/RESULTS.md`, `docs/P5_REPORT.md` and
`results/final/` (release v1.0).
