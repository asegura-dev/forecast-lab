# STATUS 2026-09-02 - The rebuild against the notebooks it corrects, cell by cell

- **Question:** this repository has claimed since its first commit to re-engineer a postgraduate notebook. Does it cover what that notebook did, and does it describe it fairly?
- **Verdict:** **Mostly, with five published errors and ten analyses that have no counterpart.** Every number this project quotes from the original checks out. But it audited the wrong notebook's indicator function, blamed the wrong mechanism for a row count, misattributed a multiplicity argument by a factor of two and a half, and - most seriously - **attributed to the notebooks a triumphalist conclusion they do not contain.** The notebook's own final cell says the model *does not* reach usable performance and that an AUC near 0.5 is consistent with efficient markets, which is close to this project's own finding.
- **Method:** both notebooks extracted cell by cell and read in full, including outputs. Every claim below verified against the file rather than against this repository's prose.
- **Sources:** `Proyecto_Final_Completo.ipynb` (41 cells: 19 code, 22 markdown, 11 figures) and `Proyecto_ASM.ipynb` (35 cells: 14 code, 21 markdown, 16 figures), plus `Documentacion_Tecnica_Codigo.docx` and `Guion_Presentacion.docx` in the same folder. Cell references are **C*n*** and **A*n*** respectively.

## 1. The counts, verified rather than repeated

| Question | Answer |
|---|---:|
| Indicator columns per symbol, `Proyecto_ASM` (A11) | **23** |
| Indicator columns per symbol, `Proyecto_Final_Completo` (C25) | **23**, a *different* 23 |
| Indicator columns in this rebuild | **19** |
| Model configurations, `Completo` (C35) | **12** |
| Model configurations, `ASM` (A32) | **30** |
| Model configurations here | **18** |
| Hypothesis tests actually run in the notebooks | **4** - one normality, three t-tests, all in `Completo` |
| ADF / KPSS / stationarity tests in the notebooks | **0** |
| Figures | 27 across both notebooks; **16** committed here |

The 23 is arithmetic, not assertion: 3 SMA + 2 EMA + 3 MACD + 1 ADX + 1 RSI + 1 ROC + 4 Bollinger + 1 ATR + returns + log returns + 3 Dist_SMA + 2 Dist_EMA. The notebook's own markdown (A12) claims **27** and its sub-counts only reach that by listing "11 features derivados" where there are 7. **The notebook miscounts its own features; this repository's 23 was right.**

The gap from 23 to 19 is policy, not omission: the raw SMA and EMA levels and the three Bollinger bands are price levels, which [ADR-006](../adr/ADR-006-features-and-stationarity.md) forbids. The distances that replace them were **already in the notebook** - `Dist_SMA200`, `Dist_EMA12` and their siblings. The original computed the scale-free version and kept the levels beside them.

## 2. The five errors in what this repository published

### 2.1 The wrong mechanism for the two row counts

[STATUS-2026-08-features](STATUS-2026-08-features.md) sec. 1 says the original's two modes did not share an index because *"its WHOLE included crypto, which begins a year after everything else"*.

**`Proyecto_ASM` excludes BTCUSD explicitly.** A1 has it commented out with the reason *"eliminado porque pierde un año de datos"*, and the load output lists ten symbols with no crypto.

The real mechanism is sharper. `indicator_creator` (A11) ends with `model.dropna()` and is **called once per symbol in a loop**, so the 199-row warm-up of the 200-period moving average is paid ten times over:

    WHOLE:  24,431 - 10 x 199 = 22,441   exactly
    FOCUS:  24,431 -      199 = 24,232   exactly

We diagnosed a real defect and named the wrong cause. The true one is a better example of the thing this project is about: a `dropna()` inside a loop, invisible at the call site, costing 1,791 rows.

### 2.2 A multiplicity argument attributed to the wrong notebook

[ADR-007](../adr/ADR-007-fitting-models-without-leaking.md) and [STATUS-2026-08-models](STATUS-2026-08-models.md) apply *"the maximum of thirty draws under the null"* to `Completo`'s `Test_AUC.idxmax()`. That `results_df` has **12 rows** - six models on two representations. Thirty is `ASM`'s count (5 x 3 x 2). The argument survives; the number does not belong where it was put.

### 2.3 The headline quote is not from the notebooks

[FINDINGS](../../FINDINGS.md) and the README attribute *"it is possible to build ML models that beat random"* to *"a postgraduate notebook"*. It is from **`Documentacion_Tecnica_Codigo.docx` sec. 12.4**, and the presentation script goes further: *"logramos construir un modelo funcional que supera el azar, alcanzando 52.1% de accuracy"*.

**The notebook's own conclusion is markedly more hedged.** C40, its last cell:

> *"Aunque el modelo **NO alcanza** niveles de rendimiento suficientes para trading automatizado (AUC > 0.7 sería deseable)... Los resultados modestos (AUC ~0.5) son **consistentes con la teoría de mercados eficientes** y la evidencia empírica en finanzas cuantitativas."*

And: *"**NO se recomienda** usar el modelo como única base para decisiones de trading."*

That is close to this project's own finding. **"The original's conclusion was wrong" is fair against the report and the presentation, and unfair against the notebook** - and this repository never distinguished the three documents. `Proyecto_ASM` states no numeric conclusion at all; its final cell is a rubric.

### 2.4 The estimator roster is attributed to one notebook and described as both

`research/models/catalogue.py` says *"The five estimators come from the original project"*. True of `ASM`. `Completo` fits **six**, including a `GradientBoostingClassifier(n_estimators=100, random_state=42)` this rebuild does not carry. `HistGradientBoostingClassifier` is marked `added_here=True` and is a different estimator with different hyperparameters; nothing says it stands in for the original's.

### 2.5 Realised volatility was not new

ADR-002 and the CHANGELOG present `realised_vol_24` as the substitute for the dropped VIX. `Completo` C25 already computed `RealVol_24h` by the same definition. It arrived here independently, which is not the same as arriving first.

## 3. What the notebooks got wrong that we never reported

The purest examples of this project's own thesis - the number that refutes a claim printed on the same screen as the claim - and we found four more than we published. All in C40, the concluding cell of `Proyecto_Final_Completo`:

| The conclusion says | The notebook's own output says |
|---|---|
| *"Mejor Modelo: **Gradient Boosting** con datos originales"* | The code selected **LightGBM (PCA)** |
| *"AUC-ROC: ~0.506"* | The selected model's AUC is **0.5104** |
| *"Oro vs Volatilidad (VIX): correlación positiva, el oro actúa como refugio"* | C27 prints **-0.173**, and the branch that fired said *"correlación débil"* |
| *"Indicadores Técnicos: RSI, MACD y medias móviles muestran **poder predictivo**"* | C27's strongest correlation with direction is **0.0285** |

## 4. Analyses with no counterpart here

Ranked by what their absence costs:

1. **`analyze_feature_scales` (A26), entirely.** Per-feature mean, std, min, max and range; the max/min range-disparity ratio that triggered the notebook's *"estandarización obligatoria"* verdict; a four-way taxonomy of features by scale; and a four-panel figure. This project's `probe_scale` asks *"does this column carry a price level?"* - a different question.
2. **The PC1-vs-PC2 scatter coloured by target (A28).** The most direct visual evidence that the classes do not separate, which is this project's whole thesis.
3. **WHOLE against FOCUS (A32, A33).** `--mode whole` exists and `features` exercises it, but **no committed model run, payload or figure uses it**, so `ASM`'s stated research question is unanswered here.
4. **The size of the move by direction (A16).** UP mean +$3.29 (sigma 4.05) against DOWN -$2.91 (sigma 4.35) - the asymmetry that explains a rising price with more falling bars. We report direction rates and never magnitudes.
5. **Train-block metrics.** Suppressed deliberately (`cli.py`: *"train accuracy measures memorisation, not skill"*), which leaves the notebooks' clearest evidence of overfitting - Random Forest at **1.0000** on train against 0.4790 on test - unrepresented, with the reasoning in a code comment rather than a decision.
6. **The full pairwise asset correlation heatmap (C27, A19).** Only target-against-symbol survives, so SPX-VIX and DXY-EURUSD are no longer visible.
7. **F1 and NPV.** Absent from the metric set, though the README quotes a classification report containing both.
8. **`Completo`'s indicator function (C25) was never mapped.** Its SMA and EMA windows of 7, 14 and 21, its five multi-lag returns (1, 3, 6, 12, 24 hours) and `MOM_10` were never kept, dropped, or recorded as either.
9. **The UP-vs-DOWN distribution figure (C23)** and the class-balance pie chart (C19).
10. **The per-symbol descriptive table with coefficient of variation (C13).** `explore` describes the target only.

## 5. What this rebuild adds that neither notebook has

Verified absent from both by keyword census: `cost`, `spread`, `power`, `bootstrap`, `adfuller`, `kpss`, `seasonal`, `hour` all return zero.

Transaction costs from the venue's own spread; a break-even per model from its measured turnover; power analysis and the minimum detectable effect; walk-forward validation over 40,587 bars; Pesaran-Timmermann against independence with a Holm correction; Hansen SPA, Romano-Wolf StepM and the Deflated Sharpe; strategy returns net of costs against an always-long benchmark; a serial-dependence measurement; three named baselines scored on every block; the gap finding; correlations on returns beside levels; ADF and KPSS reported and deliberately not used as a gate; a scale-freedom gate by rescaling; SHA-256 provenance with `verify`; a second independent dataset; and the architecture, gates and dashboard around all of it.

## 6. Is FINDINGS fair?

**Its verdict is well supported and every checkable number in it is correct.** Each row of its "What happened to the original analysis" table was verified against the notebook output: 51.53% accuracy, the 51.86% always-UP baseline it lost to, `Test_AUC.idxmax()` as literal code, the +0.923 gold-S&P correlation, and the 1,250 fabricated bars.

What it gets wrong is **attribution**, not arithmetic. It reads the surrounding report's conclusion back onto the notebook, inherits one wrong mechanism and one misplaced count into the ADR layer, and leaves four internal contradictions on the table - the four that would have made its own case best.
