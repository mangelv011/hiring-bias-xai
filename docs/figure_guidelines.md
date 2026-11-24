# Figure guidelines

Every figure in this project shows one result a reader gets at first glance. The style lives in `src/hiring_bias_xai/plotting.py`; `scripts/figure_f1.py` is the reference figure. Terms follow [GLOSSARY.md](../GLOSSARY.md).

## Rules

1. **One message per figure.** The title is that message as a full sentence, with the numbers it quotes taken from `results/*.json`. A label such as "TPR gap by occupation" is not a title. Source: Rougier et al. (2014), "Identify your message"; Wilke, "Titles, captions, and tables".
2. **Position before colour or area.** Encode values as dots, bars and lines. Use no pies and no 3D, and no heatmap when a position encoding works. Source: Cleveland and McGill (1984), whose ranking puts position along a common scale first and angle, area and colour last; Wilke, "Directory of visualizations".
3. **Everything grey, one highlight colour for what matters.** The reader's eye goes to the highlight. Source: Wilke, "Common pitfalls of color use"; Rougier et al. (2014), "Use color effectively".
4. **Direct labels instead of legends.** Put the name next to the data it names. Label only what the message needs. Source: Wilke, "Redundant coding" (direct labelling in place of legends); Rougier et al. (2014), "Do not trust the defaults".
5. **Sort by value, not alphabetically.** Ordered categories show the pattern; alphabetical order shows the alphabet. Source: Wilke, "Visualizing amounts"; Cleveland and McGill (1984) on judging positions along a common scale.
6. **Always show uncertainty and reference lines.** Draw the 95% interval of every estimate and a line at zero or at chance. Source: Wilke, "Visualizing uncertainty"; Rougier et al. (2014), "Do not mislead the reader".
7. **Colour-blind safe palette, stable meaning.** Colours are the Okabe-Ito set, and a colour means the same thing in every figure: women, men, highlight, neutral grey (`plotting.WOMEN`, `MEN`, `HIGHLIGHT`, `NEUTRAL`). Check each figure under a deuteranopia simulation (`scripts/check_deuteranopia.py`) and keep the result in `docs/figure_checks/`. Source: Okabe and Ito (2008); Wilke, "Common pitfalls of color use".
8. **No chartjunk.** No top or right spine, no grid unless it carries information, no decoration. Figures are 3.03 inches wide (one report column) and no text is smaller than 7 pt, so they stay readable at the size they print. Source: Rougier et al. (2014), "Avoid chartjunk" and "Adapt the figure to the support medium" (the 7 pt floor); Wilke, "Balance the data and the context".

## Checks

- `tests/test_figure_f1.py` asserts the width, the font size floor, the title numbers and the labels of F1.
- `scripts/check_deuteranopia.py in.png out.png` writes the deuteranope view of a figure; look at it before opening the PR.

## References

- Rougier, N. P., Droettboom, M. and Bourne, P. E. (2014). Ten Simple Rules for Better Figures. PLoS Computational Biology 10(9): e1003833.
- Wilke, C. O. (2019). Fundamentals of Data Visualization. O'Reilly. https://clauswilke.com/dataviz/
- Cleveland, W. S. and McGill, R. (1984). Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods. Journal of the American Statistical Association 79(387): 531-554.
- Okabe, M. and Ito, K. (2008). Color Universal Design (CUD): How to make figures and presentations that are friendly to colorblind people. https://jfly.uni-koeln.de/color/
- Machado, G. M., Oliveira, M. M. and Fernandes, L. A. F. (2009). A Physiologically-based Model for Simulation of Color Vision Deficiency. IEEE Transactions on Visualization and Computer Graphics 15(6): 1291-1298.
