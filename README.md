# TS-TrajGAN


## Install the dependency

`./requirements.txt`

## Dataset

`./data/`

## Run the models

`run syn_pred_data.py`


## Output files

### Model training parameters stored under the directory

`./out_data_path/One_Stage_Model/`
`./out_data_path/Two_Stage_Model/`

### Generated data under the directory 

`./out_data_path/syn_data_output/`: 

`formalized_txt_syn?_pred?_min?_max?.txt`: represents the generated trajectory dataset without time information

`formalized_txt_syn?_pred?_min?_max?_with_time.txt`: represents the generated trajectory dataset with time information


### Run the Evaluation

`run make_metrics_and_vis.py`

`You need to specify the path of the original trajectory dataset 'ori_traj_txt_path' and the generated trajectory dataset 'syn_traj_txt_path' under the main function, as well as the output path of the evaluation results`

`1-pca.png and 1-tsne.png are result of two stage generated data`

`metrics_score.txt recordeds Discriminative Scores and Predictived Score for data generated in two stages`


