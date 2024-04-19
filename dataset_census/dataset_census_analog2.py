import pickle
import yaml
import os
import math
import re
import numpy as np
import pandas as pd
import category_encoders as ce
from torch.utils.data import DataLoader, Dataset
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# import required packages
import torch
# import wget
# wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')
from utils2 import *


def process_func(path: str, cat_list, encode=True, 
                 mecha1 = 'MAR', m_ratio1 = 0.2, m_cols = None, 
                 mecha2 = 'MAR', m_ratio2 = 0.2, opt="selfmasked"):
    def sortby(x):
        if isinstance(x, str):
            return ord(x[1])
        else:
            return x

    def check_col(x):
        if "_" in x:
            return False
        else:
            return True

    data = pd.read_csv(path, header=None)

    data.replace(0, 1, inplace=True)
    data.replace(" ?", np.nan, inplace=True)
    ###
    from sklearn.impute import SimpleImputer
    my_imputer = SimpleImputer(strategy='most_frequent')
    data = pd.DataFrame(my_imputer.fit_transform(data))

    ### mechanism에 따라 observed_masks를 인위적으로 생성하기 ###
    # 인코딩 과정을 앞으로 가져옴
    encoder = ce.OrdinalEncoder(cols=data.columns[cat_list]) ### ce.ordinal.OrdinalEncoder
    data_enc = encoder.fit_transform(data) # 범주형 변수를 순서형 변수로 인코딩
    data_observed_values = data_enc.values # 마스킹 없는 complete 데이터를 array로
    # observed_masks 생성
    observed_masks = produce_NA(data_observed_values, mecha=mecha1, opt=opt, m_ratio=m_ratio1, m_cols=m_cols) # missing:1
    observed_masks = np.array(observed_masks) # array 처리
    observed_masks = (1-observed_masks).astype(bool)
    data2 = np.array(data, copy=True) ### observed_values 대신 data2 일단 사용
    data2[~observed_masks] = np.nan
    ###

    # observed_masks = ~pd.isnull(data)
    # observed_masks = observed_masks.values

    
    data1 = pd.DataFrame(data2).replace(np.nan, 0)

    last_bits_num = 0
    num_bits_list = []
    new_df = data1.copy() ### 0->1, nan->0으로 대체된 dataframe. 인코딩 등 되지 않음. 
    new_df.columns = new_df.columns.astype(str)
    tot_map_dict = []
    last_bits_num = 0

    ###
    # encoder = ce.ordinal.OrdinalEncoder(cols=data.columns[cat_list]) ## ordinal encoding
    # encoder.fit(data)
    new_df2 = encoder.transform(data2) # 기존 ordinal encoder 사용
    new_df2.replace(np.nan, 0, inplace=True)
    new_observed_values2 = new_df2.values ### 1차 결측치 처리 하고 ordinal encoding후 nan을 0으로 바꾼 데이터프레임의 array

    gt_masks = produce_NA(new_observed_values2, mecha=mecha2, opt=opt, m_ratio=m_ratio2, m_cols=m_cols, seed=1) ### seed 추가.
    gt_masks = (1-np.array(gt_masks)) #.astype(bool) # 1: observed, 0: missing ### 아 근데 이거... 맞나 모르겠음
    gt_masks = gt_masks * observed_masks # observed_masks에서 0인 것은 0으로 처리해 두 mask를 합침
    ###

    # new_observed_values2 = new_df2.values
    for col in cat_list:
        cat_num = data1.iloc[:, col].nunique() # 범주형 하나 안에 있는 카테고리 개수
        bits_num = int(math.log2(cat_num)) + 1 # 그 카테고리를 이진으로 표현시 필요한 컬럼 수
        num_bits_list.append(bits_num)
        map_target = [i for i in range(1, cat_num + 1)]

        unique_obj = list(data1.iloc[:, col].unique())
        # exclude 0
        unique_obj = [i for i in unique_obj if i != 0]

        unique_obj.sort(key=sortby)
        map_dict = {
            unique_obj[i]: bin(map_target[i])[2:].zfill(bits_num)
            for i in range(len(unique_obj))
        }

        # create key-value pair for missing values
        map_dict[0] = "0" * bits_num

        tot_map_dict.append(map_dict)
        data1.iloc[:, col] = data1.iloc[:, col].map(map_dict)

        unique_obj = list(data1.iloc[:, col].unique())

        # new_df.drop(col+last_bits_num, inplace=True, axis=1)
        for i in range(bits_num):
            new_df.insert(
                col + i + last_bits_num, f"{col}_{i}", data1.iloc[:, col].str[i]
            )
            new_df.iloc[:, col + i + last_bits_num] = new_df.iloc[
                :, col + i + last_bits_num
            ].astype(int)
            new_df.iloc[:, col + i + last_bits_num] = (
                2 * new_df.iloc[:, col + i + last_bits_num] - 1
            )
        last_bits_num += bits_num

    # remove original categorical columns
    new_df.drop(list(map(str, cat_list)), axis=1, inplace=True)

    new_observed_values = new_df.values
    masks = observed_masks.copy()

    ############# real data에 대해 동일한 처리 #############
    # data, real_df
    last_bits_num = 0
    num_bits_list = []
    real_df = data.copy() ### 0->1, nan->0으로 대체된 dataframe. 인코딩 등 되지 않음. 
    real_df.columns = real_df.columns.astype(str)
    tot_map_dict = []
    last_bits_num = 0
    
    for col in cat_list:
        cat_num = data.iloc[:, col].nunique() # 범주형 하나 안에 있는 카테고리 개수
        bits_num = int(math.log2(cat_num)) + 1 # 그 카테고리를 이진으로 표현시 필요한 컬럼 수
        num_bits_list.append(bits_num)
        map_target = [i for i in range(1, cat_num + 1)]

        unique_obj = list(data.iloc[:, col].unique())
        # exclude 0
        unique_obj = [i for i in unique_obj if i != 0]

        unique_obj.sort(key=sortby)
        map_dict = {
            unique_obj[i]: bin(map_target[i])[2:].zfill(bits_num)
            for i in range(len(unique_obj))
        }

        # create key-value pair for missing values
        map_dict[0] = "0" * bits_num

        tot_map_dict.append(map_dict)
        data.iloc[:, col] = data.iloc[:, col].map(map_dict)

        unique_obj = list(data.iloc[:, col].unique())

        # new_df.drop(col+last_bits_num, inplace=True, axis=1)
        for i in range(bits_num):
            real_df.insert(
                col + i + last_bits_num, f"{col}_{i}", data.iloc[:, col].str[i]
            )
            real_df.iloc[:, col + i + last_bits_num] = real_df.iloc[
                :, col + i + last_bits_num
            ].astype(int)
            real_df.iloc[:, col + i + last_bits_num] = (
                2 * real_df.iloc[:, col + i + last_bits_num] - 1
            )
        last_bits_num += bits_num

    # remove original categorical columns
    real_df.drop(list(map(str, cat_list)), axis=1, inplace=True)

    real_values = real_df.values

    #############################
    
    cum_num_bits = 0
    new_observed_masks = observed_masks.copy()
    new_gt_masks = gt_masks.copy()

    for index, col in enumerate(cat_list):
        add_col_num = num_bits_list[index]
        insert_col_obs = observed_masks[:, col]
        insert_col_gt = gt_masks[:, col]

        for i in range(add_col_num - 1):
            new_observed_masks = np.insert(
                new_observed_masks, cum_num_bits + col, insert_col_obs, axis=1
            )
            new_gt_masks = np.insert(
                new_gt_masks, cum_num_bits + col, insert_col_gt, axis=1
            )
        cum_num_bits += add_col_num - 1

    # get columns for continous variables
    cont_cols = []
    for index, col_name in enumerate(new_df.columns):
        if check_col(col_name):
            cont_cols.append(index)

    saved_cat_cols = {}
    for index, col in enumerate(cat_list):
        indices = [
            i for i, s in enumerate(new_df.columns) if s.startswith(str(col) + "_")
        ]
        saved_cat_cols[str(index)] = indices

    with open("./data_census_analog/transform.pk", "wb") as f:
        pickle.dump([tot_map_dict, cont_cols, saved_cat_cols], f)

    # NaN is replaced by zero
    new_observed_values = np.nan_to_num(new_observed_values)
    new_observed_values = new_observed_values.astype(float)

    ### for real values
    real_values = np.nan_to_num(real_values)
    real_values = real_values.astype(float)

    # observed_masks: 0 for missing elements
    observed_masks = observed_masks.astype(int)  # "float32"
    gt_masks = gt_masks.astype(int)


    return real_values, new_observed_values, new_observed_masks, new_gt_masks, cont_cols


class tabular_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(self, eval_length=39, use_index_list=None, seed=0,
                 mecha1 = 'MAR', m_ratio1 = 0.2, m_cols=[0],
                 mecha2 = 'MAR', m_ratio2 = 0.2, opt="selfmasked"):
        self.eval_length = eval_length
        np.random.seed(seed)

        col_type = 'missing_cat' if m_cols ==[1, 3, 5, 6, 7, 8, 9, 13, 14] else 'missing_num' if m_cols == [0,2,4,10,11,12] else 'random'

        dataset_path = "./data_census_analog/adult_trim.data"
        processed_data_path = (
            f"./data_census_analog/{mecha1}-{mecha2}_seed-{seed}_{col_type}.pk"
        )
        processed_data_path_norm = f"./data_census_analog/{mecha1}-{mecha2}_seed-{seed}_{col_type}_max-min_norm.pk"

        cat_list = [1, 3, 5, 6, 7, 8, 9, 13, 14]
        if not os.path.isfile(processed_data_path):
            (
                self.real_values,
                self.observed_values,
                self.observed_masks,
                self.gt_masks,
                self.cont_cols,
            ) = process_func(
                dataset_path,
                cat_list=cat_list,
                mecha1=mecha1, m_ratio1=m_ratio1, m_cols=m_cols,
                mecha2=mecha2, m_ratio2=m_ratio2,
                opt=opt,
                encode=True,
            ) ###

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [
                        self.real_values,
                        self.observed_values,
                        self.observed_masks,
                        self.gt_masks,
                        self.cont_cols,
                    ],
                    f,
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):  # load datasetfile
            with open(processed_data_path_norm, "rb") as f:
                self.real_values, self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "real_data": self.real_values[index],
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, 
                   m_ratio1=0.2, mecha1='MAR', m_cols=[0],
                   m_ratio2=0.2, mecha2='MAR', opt='selfmasked'): ###
    dataset = tabular_dataset(m_ratio1=m_ratio1, mecha1=mecha1, m_cols=m_cols,
                              m_ratio2=m_ratio2, mecha2=mecha2, opt=opt, seed=seed) ###
    
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    full_index = indlist ### 

    num_train = (int)(len(indlist) * 0.80) ### valid set도 만들기 위해 조정.
    ### 결과적으로 train:valid:test = 4:1:0 = 16000:4000:0
    train_index = indlist[:num_train]
    valid_index = indlist[num_train:]

    # Here we perform max-min normalization.
    col_type = 'missing_cat' if m_cols ==[1, 3, 5, 6, 7, 8, 9, 13, 14] else 'missing_num' if m_cols == [0,2,4,10,11,12] else 'random'
    processed_data_path_norm = f"./data_census_analog/{mecha1}-{mecha2}_seed-{seed}_{col_type}_max-min_norm.pk"
    if not os.path.isfile(processed_data_path_norm):
        print(
            "--------------Dataset has not been normalized yet. Perform data normalization and store the mean value of each column.--------------"
        )
        # Data transformation after train-test split.
        col_num = len(dataset.cont_cols)
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        for index, k in enumerate(dataset.cont_cols):
            # Using observed_mask to avoid counting missing values (now represented as 0)
            obs_ind = dataset.observed_masks[train_index, k].astype(bool)
            temp = dataset.observed_values[train_index, k]
            max_arr[index] = max(temp[obs_ind])
            min_arr[index] = min(temp[obs_ind])
        print(
            f"--------------Max-value for cont-variable column {max_arr}--------------"
        )
        print(
            f"--------------Min-value for cont-variable column {min_arr}--------------"
        )

        for index, k in enumerate(dataset.cont_cols):
            dataset.observed_values[:, k] = (
                (dataset.observed_values[:, k] - (min_arr[index] - 1))
                / (max_arr[index] - min_arr[index] + 1)
            ) * dataset.observed_masks[:, k]

        ### for real data
        for index, k in enumerate(dataset.cont_cols):
            dataset.real_values[:, k] = (
                (dataset.real_values[:, k] - (min_arr[index] - 1))
                / (max_arr[index] - min_arr[index] + 1)
            )
        ###

        with open(processed_data_path_norm, "wb") as f:
            pickle.dump(
                [dataset.real_values, dataset.observed_values, dataset.observed_masks, dataset.gt_masks], f
            )

    # Create datasets and corresponding data loaders objects.
    full_dataset = tabular_dataset( ## 전체 데이터셋을 가져오는 로더
        use_index_list=full_index, mecha1=mecha1, mecha2=mecha2, m_cols=m_cols, seed=seed ### missing_ratio(2) 대신 mecha1, mecha2 를 넣어야함 (path에 들어가는 요소!!!)
    )
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=0)

    train_dataset = tabular_dataset(
        use_index_list=train_index, mecha1=mecha1, mecha2=mecha2, m_cols=m_cols, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = tabular_dataset(
        use_index_list=valid_index, mecha1=mecha1, mecha2=mecha2, m_cols=m_cols, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    # test_dataset = tabular_dataset(
    #     use_index_list=test_index, mecha1=mecha1, mecha2=mecha2, seed=seed
    # )
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    # print(f"Testing dataset size: {len(test_dataset)}")

    return full_loader, train_loader, valid_loader #test_loader
