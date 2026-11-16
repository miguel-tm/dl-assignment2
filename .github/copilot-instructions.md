# Copilot instructions: dl-assignment2

SCS 3546 Assignment 2: VGG16 transfer learning on `flower_photos` (5 classes), in one notebook:
`notebooks/SCS_3546_Assignment_02.ipynb`.

## Read first
1. `STATUS.md`: what is done, current phase, environment quirks, per-phase walkthrough.
2. `PLAN.md`: phase checklist; source of truth for what to do.
3. `DECISIONS.md`: decision log (IDs like M01, Q1, D01, E09); source of truth for why. Cite IDs when relevant.
4. `dev/handoffs/`: the prompt for the current phase, if one exists.
5. `README.md`: environment setup only (WSL venv, kernel, Colab).

If these conflict, PLAN.md and DECISIONS.md win; flag the conflict instead of guessing.

## Working rules
- Markdown cells: short headings and `[TODO Miguel: ...]` placeholders only. Miguel writes all analysis.
- Code: small functions with short docstrings, clear comments. Helpers live in notebook cells under "Helper Functions", not in a module (M04).
- No hardcoded Windows, `/content/` or Google Drive paths; must run on WSL and Colab.
- Work in small steps; after each, say what to run and what output to expect.
- Ask when something is ambiguous.
- Do not commit, push, or install/upgrade packages; Miguel does that.
- At the end of a phase, add a section to `STATUS.md` and update its table.
