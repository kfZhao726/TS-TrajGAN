# TS-TrajGAN

《Generating Spatiotemporal Trajectories with GANs and Conditional GANs》

## Install the dependency

`./requirements.txt`

## Dataset

`./data/`

## Run the models

`run syn_pred_data.py`

`可以在主函数中修改参数。比如数据集路径，输出文件路径；一阶段生成轨迹的长度，二阶段预测轨迹的最大长度；模型的隐层维度等`


## Output files

### Model training parameters stored under the directory

`./out_data_path/One_Stage_Model/` `./out_data_path/Two_Stage_Model/`

`第一次训练得到的模型参数会保存到本地，如果想继续训练或利用这些参数进行生成的话，可以直接读取;`

`如果要从头训练或训练其它数据集，请保证该文件夹是空目录`

### Generated datas under the directory 

`./out_data_path/syn_data_output/`: 

`formalized_txt_syn?_pred?_min?_max?.txt`: represents the generated trajectory dataset without time information

`formalized_txt_syn?_pred?_min?_max?_with_time.txt`: represents the generated trajectory dataset with time information


### Run the Evaluation

`run make_metrics_and_vis.py`

`您需要在main函数下指定原始轨迹数据集和生成轨迹数据集(以with_time结尾的txt文件)的路径，以及评估结果的输出路径`

`You need to specify the path of the original trajectory dataset 'ori_traj_txt_path' and the generated trajectory dataset 'syn_traj_txt_path' under the main function, as well as the output path of the evaluation results`

`1-pca.png and 1-tsne.png are result of two stage generated data`

`metrics_score.txt recordeds Discriminative Scores and Predictived Score for data generated in two stages`


