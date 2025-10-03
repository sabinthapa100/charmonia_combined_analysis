Charmonium Suppression Analysis Framework

A Python framework to compute the nuclear modification factor, R 
pA
​
 , by combining models for Cold Nuclear Matter (CNM) and primordial state suppression, including robust error propagation and visualization tools.

Theoretical Framework

The nuclear modification factor, $R_pA$, is calculated by factorizing contributions from CNM effects (nPDFs and energy loss) and primordial suppression:

R_{pA}^{\text{Total}} = \underbrace{\left( R_{pA}^{\text{nPDF}} \times R_{pA}^{\text{E-loss}} \right)}_{\text{CNM}} \times R_{pA}^{\text{Primordial}} $$Asymmetric uncertainties ($R_c^{+\Delta_+}_{-\Delta_-}$) are propagated by adding the relative errors ($\delta_{\pm} = \Delta_{\pm}/R_c$) in quadrature for products:  $$\delta\_{\text{product},\pm} = \sqrt{\sum\_i (\delta\_{i,\pm})^2} $$\#\# Quickstart  The analysis is driven by the `Combiner` class. The directory structure is expected as follows:  ```bash . |-- input/ |-- code/ `-- analysis_notebook.ipynb ```  



### Usage Example  Instantiate a `Combiner` for each primordial model (e.g., NPWLC and Perturbative) to compare results.  ```python from code.combine_module import Combiner  # 1. Define paths to input data ELOSS_5TEV = "./input/eloss/pPb5TeV" NPDF_5TEV  = "./input/npdf/pPb5TeV" GLAUBER_5TEV = "./input/glauber_data/5TeV" PRIM_BASE_NPWLC = "./input/primordial/pPb5TeV/output_5pPb_Tf170_NPWLC" PRIM_BASE_PERT  = "./input/primordial/pPb5TeV/output_5pPb_Tf170_Pert"  # 2. Initialize Combiner for each model c5_npwlc = Combiner(     tag="5.02", prim_base=PRIM_BASE_NPWLC, e_loss_base=ELOSS_5TEV,     npdf_folder=NPDF_5TEV, glauber_root=GLAUBER_5TEV,     sigmaNN_mb=67.0, sqrt_sNN_GeV=5020.0 )  c5_pert = Combiner(     tag="5.02", prim_base=PRIM_BASE_PERT, e_loss_base=ELOSS_5TEV,     npdf_folder=NPDF_5TEV, glauber_root=GLAUBER_5TEV,     sigmaNN_mb=67.0, sqrt_sNN_GeV=5020.0 )  # 3. Use plotting functions to generate results # (See notebook for helper functions to overlay model comparisons) # e.g., compare_total_vs_centrality([c5_npwlc, c5_pert], ...) ```
