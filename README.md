### Local Point-Wise Explanations of LambdaMART


In order to replicate the experiments, you can train LambdaMART in Local Point-Wise Explanations by looking into ```Get exp (yahoo and web10k).ipynb``` or load the ones already trained in `/models`.

Run ```get_exp_all_datasets -d DATASET_NAME``` to obtain explanations (or load the ones already obtained in `/exps` folder. 

Run ```Evaluation Pointwise (V2).ipynb``` replicating the evaluation results and ```Evaluation Viz.ipynb``` for replicating the visualizations.

Lastly, run ```consistency/Consinstency.ipynb``` to replicate our resutls for Explanation consistency and ```investigation/Investigation.ipynb``` for the effect of depth. 