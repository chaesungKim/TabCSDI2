import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch

import pickle
import yaml
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
          return super().find_class(module, name)
        
from src.main_model_table import TabCSDI
# from src.utils_table import train, evaluate_analog_all
# from dataset_insurance.dataset_insurance_analog2 import get_dataloader

# Analog, MCAR
# bringing generation results
def open_results_an(name, mecha1, mecha2, m_type, m_ratio1, m_ratio2, nsample, seed, start_ind=0, end_ind=20000): # m_ratio1 추가, name 추가
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(f"./save/{name}_analog_fold5/m_ratio1_{m_ratio1}/{mecha1}{mecha2}_{m_type}/full_generated_outputs_nsample{nsample}seed_{seed}.pk", 'rb') as f :
        if device == "cpu" : results = CPU_Unpickler(f).load()
        else : results = pickle.load(f)

    # modelfolder = path_model #
    config = "census_onehot_analog.yaml"
    # exe_name = name

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    path = "config/" + config

    if name == "insurance":
        m_cols = [1, 4, 5] if m_type == 'missing_cat' else [0,2,3,6] if m_type == 'missing_num' else [0,1,4,6]

    elif name == "census":
        m_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14] if m_type == 'missing_cat' else [0,2,4,10,11,12] if m_type == 'missing_num' else [1,2,3,4,6,7,8,11,12,13]

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = 0
    config["model"]["m_ratio2"] = m_ratio2
    ###
    config["model"]["mecha2"] = mecha2 ##
    config["model"]["m_cols"] = m_cols ##
    ##
    config["model"]["name"] = name ## 

    model = TabCSDI(config, device).to(device)

    model.load_state_dict(torch.load(f"./save/{name}_analog_fold5/m_ratio1_{m_ratio1}/{mecha1}{mecha2}_{m_type}/model_{seed}.pth", map_location=device))
    model.eval()

    fsamples = results[0] #
    fsamples_median = fsamples.median(dim=1)
    fc_target = results[1] #
    fevalpoints = results[2] #
    fobspoints = results[3] #
    fevalpoints = fevalpoints.squeeze()
    freal = results[7]

    ### real data 불러오기
    # with open("./save/census_analog_fold5/MCARMCAR_20240406_155903/full_real_data.pk", 'rb') as f :
    #     real1 = CPU_Unpickler(f).load()
    ###


    if name == "insurance":
        with open("./data_insurance/transform.pk", "rb") as f:
            _, cont_cols, saved_cat_dict = pickle.load(f)
    elif name == "census":
        with open("./data_census_analog/transform.pk", "rb") as f:
            _, cont_cols, saved_cat_dict = pickle.load(f)

    # Threashold
    for i in range(fsamples_median.values.size(dim=1)):
      if i in cont_cols:
        continue
      index1 = fsamples_median.values[:, i, :] >= 0
      fsamples_median.values[:, i, :][index1] = 1
      index2 = fsamples_median.values[:, i, :] < 0
      fsamples_median.values[:, i, :][index2] = -1

    fsamples_median_df = pd.DataFrame(fsamples_median.values.squeeze().cpu())
    fevalpoints_df = pd.DataFrame(np.array(fevalpoints.cpu())) #
    fsamples_median_df = fsamples_median_df * fevalpoints_df #
    fc_target_df = pd.DataFrame(fc_target.squeeze().cpu())
    ftarget_imputed_df = fc_target_df.where(fevalpoints_df==0,fsamples_median_df) #
    ###
    freal_df = pd.DataFrame(freal.squeeze().cpu())

    fc_target_df = fc_target_df[start_ind:end_ind]
    fsamples_median_df = fsamples_median_df[start_ind:end_ind]
    fevalpoints_df = fevalpoints_df[start_ind:end_ind]
    ftarget_imputed_df = ftarget_imputed_df[start_ind:end_ind]
    freal_df = freal_df[start_ind:end_ind]

    return fc_target_df, fsamples_median_df, fevalpoints_df, ftarget_imputed_df, freal_df

# Analog 결과 계산 함수 - 2 (real로 비교)
def analog_results(name, fevalpoints, ftarget, fimputed, freal): # name 추가

    if name == "insurance":
        an_cont_list = [0, 3, 4, 10]
        with open("./data_insurance/transform.pk", "rb") as f:
            _, cont_cols, saved_cat_dict = pickle.load(f)
    elif name == "census":
        an_cont_list = [0, 5, 11, 28, 29, 30]
        with open("./data_census_analog/transform.pk", "rb") as f:
            _, cont_cols, saved_cat_dict = pickle.load(f)

    fevalpoints_cont = fevalpoints.iloc[:,an_cont_list]
    fdiff = (freal-fimputed).iloc[:,an_cont_list]
    fdiff = fdiff.where(fevalpoints_cont==1)
    frmse = np.sqrt((fdiff ** 2).mean())
    frmse_avg = frmse.mean()

    ferr_total = np.zeros([len(saved_cat_dict)])
    ferr_total_eval_nums = np.zeros([len(saved_cat_dict)])
    for i in range(len(saved_cat_dict)):
        cate_cols = saved_cat_dict[str(i)]
        fmatched_nums = (
            (
                (
                    fimputed.values[:, cate_cols] ###
                    == freal.values[:, cate_cols]
                )
                * fevalpoints.values[:, cate_cols]
            )
            .all(1)
            .sum()
        )
        feval_nums = fevalpoints.values[:, cate_cols].sum() / len(
            cate_cols
        )
        ferr_total[i] += feval_nums - fmatched_nums
        ferr_total_eval_nums[i] += feval_nums

    ferror = ferr_total / ferr_total_eval_nums
    # ferror_avg = np.nanmean(ferror)
    ferror_avg = ferr_total.sum() / ferr_total_eval_nums.sum()

    return frmse, frmse_avg, ferror, ferror_avg

def impute_evaluation(name, mecha1, mecha2, m_type, m_ratio1, m_ratio2, nsample, seed_list):
    rmse = pd.DataFrame()
    error = pd.DataFrame()
    for s in seed_list:
        seed = s
        ftarget, fsamples, fevalpoints, fimputed, freal = open_results_an(name, mecha1, mecha2, m_type, m_ratio1, m_ratio2, nsample, seed)
        frmse, frmse_avg, ferror, ferror_avg = analog_results(name, fevalpoints, ftarget, fimputed, freal)
        rmse[str(seed)] = frmse
        error[str(seed)] = pd.Series(ferror)
        rmse.loc['avg',str(seed)] = frmse_avg
        error.loc['avg',str(seed)] = ferror_avg

    rmse['mean'] = rmse.mean(axis=1)
    rmse['std'] = rmse.std(axis=1)
    error['mean'] = error.mean(axis=1)
    error['std'] = error.std(axis=1)

    return rmse, error