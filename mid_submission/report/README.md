# Mid-submission report: experiments section

Content for the TODO placeholders of Sections 5.1, 5.2 and 6 of the report (`../ANLP_final_proposal-6.pdf`).

| File | Use |
|---|---|
| `experiments.tex` | LaTeX for §5.1 Datasets, §5.2 Model and Training, §6 Results (Phase I study + RL baselines) |
| `figures/*.pdf` | The two figures (vector); `.png` copies for slides |
| `extra_refs.bib` | BibTeX for the new citations (TreeRL, DeepSeekMath/GRPO, MATH, PRM800K, Omni-MATH, Qwen2.5-Math) |
| `preview.tex`, `preview.pdf` | Local preview in the ACL page size (compile with `xelatex`) |

**To add it to the Overleaf report**
1. Upload `figures/` next to the main `.tex` and add `extra_refs.bib` to the bibliography (or merge the entries).
2. Make sure the preamble has `\usepackage{graphicx}` and `\usepackage{booktabs}`.
3. Replace the TODO text of §5.1, §5.2 and §6 with the corresponding parts of `experiments.tex` (or `\input{experiments}` in place of §5.1 to §6 and delete the old headings).
4. If your `.bib` already has TreeRL under another key, rename `hou2025treerl` in `experiments.tex`.

To redraw the figures and print the numbers: `python -m tree_alloc.report_figures` from `mid_submission/`.
