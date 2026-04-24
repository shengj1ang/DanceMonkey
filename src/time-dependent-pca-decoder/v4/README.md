PCA Sweep Experiments

The following scripts are used to evaluate the effect of PCA dimensionality on decoding performance:

* runPcaDimSweep.m
    Sweeps over fixed PCA dimensions and evaluates RMSE.
* runPcaVarKeepSweep.m
    Sweeps over variance-retention thresholds (varKeep) and automatically selects PCA dimensions.

Outputs

Running these scripts generates:

* .mat files containing experiment results
* .csv files for further analysis
* Plots such as:
    * RMSE-vs-Mean-Selected-PCA-Dimension.png

These experiments were used to identify the optimal PCA dimensionality (around 95% variance retention), which significantly improves decoding performance.
