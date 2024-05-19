### Local Point-Wise Explanations of LambdaMART
This is the code repository for the paper, "Local Pointwise Explanations of LambdaMART" accepted at [The 14th Scandinavian conference on AI (SCAI)](https://ju.se/samarbeta/event-och-konferenser/konferens/scai-symposium-2024.html). 

In order to replicate the experiments, you can train LambdaMART in Local Point-Wise Explanations by looking into ```Get exp (yahoo and web10k).ipynb``` or load the ones already trained in `/models`.

Run ```get_exp_all_datasets -d DATASET_NAME``` to obtain explanations (or load the ones already obtained in `/exps` folder. 

Run ```Evaluation Pointwise (V2).ipynb``` replicating the evaluation results and ```Evaluation Viz.ipynb``` for replicating the visualizations.

Lastly, run ```consistency/Consinstency.ipynb``` to replicate our resutls for Explanation consistency and ```investigation/Investigation.ipynb``` for the effect of depth. 
