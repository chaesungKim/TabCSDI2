# This script is for generating .pk file for mixed data types dataset
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
from utils import *

# Function produce_NA for generating missing values

def produce_NA(X, p_miss, mecha="MAR", opt=None, p_obs=None, q=None):
    
    to_torch = torch.is_tensor(X)
    
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    np.random.seed(0);torch.manual_seed(0)
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    # X_nas = X.clone()
    # X_nas[mask.bool()] = np.nan
    
    # return X_nas.double() # tensor type
    return mask

# process_func: masking을 생성하고, 인코딩 방식에 따라 인코딩하는 함수
# return: observed_values, observed_masks, gt_masks, cont_list
def process_func(path: str, cat_list, missing_ratio=0.4, encode=True, p_obs=0.5, mecha='MAR', opt="logistic"):

    data = pd.read_csv(path, header=None)

    # Swap columns
    temp_list = [i for i in range(data.shape[1]) if i not in cat_list]
    temp_list.extend(cat_list)
    new_cols_order = temp_list
    data = data.reindex(columns=data.columns[new_cols_order])
    data.columns = [i for i in range(data.shape[1])]

    # create two lists to store position
    cont_list = [i for i in range(0, data.shape[1] - len(cat_list))]
    cat_list = [i for i in range(len(cont_list), data.shape[1])]

    observed_values = data.values
    observed_masks = ~pd.isnull(data) # masking 생성 전. 결측이면 False, 값이 있으면 True
    observed_masks = observed_masks.values # dataframe -> np.array

    # masks = observed_masks.copy()
    # # In this section, obtain gt_masks
    # # for each column, mask `missing_ratio` % of observed values.
    # for col in range(masks.shape[1]):
    #     obs_indices = np.where(masks[:, col])[0]
    #     miss_indices = np.random.choice(
    #         obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    #     )
    #     masks[miss_indices, col] = False
    # # gt_mask: 0 for missing elements and manully maksed elements
    # gt_masks = masks.reshape(observed_masks.shape)

    encoder = ce.ordinal.OrdinalEncoder(cols=data.columns[cat_list])
    encoder.fit(data)
    new_df = encoder.transform(data)
    new_observed_values = new_df.values
    print(new_observed_values)

    gt_masks = produce_NA(new_observed_values, p_miss=missing_ratio, p_obs=p_obs, mecha=mecha, opt=opt)
    gt_masks = np.array(gt_masks)

    num_cate_list = []
    if encode == True:
        # set encoder here
        ## masking 전으로 옮김
        # encoder = ce.ordinal.OrdinalEncoder(cols=data.columns[cat_list])
        # encoder.fit(data)
        # new_df = encoder.transform(data)
        # we now need to transform these masks to the new one, suitable for mixed data types.
        cum_num_bits = 0
        new_observed_masks = observed_masks.copy()
        new_gt_masks = gt_masks.copy()

        for index, col in enumerate(cat_list):
            num_cate_list.append(new_df.iloc[:, col].nunique())
            corresponding_cols = len(
                [
                    s
                    for s in new_df.columns
                    if isinstance(s, str) and s.startswith(str(col) + "_")
                ]
            )
            add_col_num = corresponding_cols
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

        new_observed_values = new_df.values
        new_observed_values = np.nan_to_num(new_observed_values)
        new_observed_values = new_observed_values.astype(np.float)

        with open("./data_census_ft/transformed_columns.pk", "wb") as f:
            pickle.dump([cont_list, num_cate_list], f)

        with open("./data_census_ft/encoder.pk", "wb") as f:
            pickle.dump(encoder, f)

    if encode == True:
        return new_observed_values, new_observed_masks, new_gt_masks, cont_list
    else:
        cont_cols = [i for i in data.columns if i not in cat_list]
        return observed_values, observed_masks, gt_masks, cont_list


class tabular_Dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(self, eval_length=15, use_index_list=None, missing_ratio=0.4, seed=0, p_obs=0.5, mecha='MAR', opt="logistic"):
        self.eval_length = eval_length
        np.random.seed(seed)

        dataset_path = "./data_census_ft/adult_trim.data"
        processed_data_path = (
            f"./data_census_ft/missing_ratio-{missing_ratio}_seed-{seed}.pk"
        )
        processed_data_path_norm = f"./data_census_ft/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"

        # self.cont_cols is only saved in .pk file before normalization.
        cat_list = [1, 3, 5, 6, 7, 8, 9, 13, 14]
        if not os.path.isfile(processed_data_path): # process_func을 사용해서 전체 데이터를 처리 (masking & encoding)
            # 이때 처리된 데이터가 get_dataloader에서 normalized되고, train val test split 단계에서 다시한번 이 함수가 쓰이나 masking 과정은 거치지 않는다.
            (
                self.observed_values,
                self.observed_masks,
                self.gt_masks,
                self.cont_cols,
            ) = process_func(
                dataset_path,
                cat_list=cat_list,
                missing_ratio=missing_ratio,
                p_obs=p_obs, mecha=mecha, opt=opt,
                encode=True,
            )

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [
                        self.observed_values,
                        self.observed_masks,
                        self.gt_masks,
                        self.cont_cols,
                    ],
                    f,
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):  # load datasetfile
            # 앞에서 전체 데이터가 masked, encoded, normalized되고 이를 불러옴. index list로 split만 해줌. 
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values)) # 전체 데이터를 처리할 때에는, index list가 필요하지 않음
        else:
            self.use_index_list = use_index_list # train test split할 때 index를 제공

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]  # 
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.4, p_obs=0.5, mecha='MAR', opt="logistic"):
    dataset = tabular_Dataset(missing_ratio=missing_ratio, seed=seed, p_obs=p_obs, mecha=mecha, opt=opt)
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))

    np.random.seed(seed + 1)
    np.random.shuffle(indlist)

    tmp_ratio = 1 / nfold
    start = (int)((nfold - 1) * len(dataset) * tmp_ratio)
    end = (int)(nfold * len(dataset) * tmp_ratio)

    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.shuffle(remain_index)
    num_train = (int)(len(remain_index) * 1)

    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    # Here we perform max-min normalization.
    processed_data_path_norm = (
        f"./data_census_ft/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
    )
    if not os.path.isfile(processed_data_path_norm):
        print(
            "--------------Dataset has not been normalized yet. Perform data normalization and store the mean value of each column.--------------"
        )
        # data transformation after train-test split.
        col_num = len(dataset.cont_cols)
        max_arr = np.zeros(col_num)
        min_arr = np.zeros(col_num)
        mean_arr = np.zeros(col_num)
        for index, k in enumerate(dataset.cont_cols): ## 각각의 연속형 변수 컬럼에 대해 정규화 진행
            # Using observed_mask to avoid counting missing values (now represented as 0)
            obs_ind = dataset.observed_masks[train_index, k].astype(bool) # train set에서 관측된 데이터 선택
            temp = dataset.observed_values[train_index, k] # train set의 데이터 값들
            max_arr[index] = max(temp[obs_ind])
            min_arr[index] = min(temp[obs_ind])

        print(
            f"--------------Max-value for cont-variable column {max_arr}--------------"
        )
        print(
            f"--------------Min-value for cont-variable column {min_arr}--------------"
        )

        for index, k in enumerate(dataset.cont_cols): ## 각각의 연속형 변수 컬럼에 대해 정규화 진행
            dataset.observed_values[:, k] = (
                (dataset.observed_values[:, k] - (min_arr[index] - 1))
                / (max_arr[index] - min_arr[index] + 1)
            ) * dataset.observed_masks[:, k]

        with open(processed_data_path_norm, "wb") as f:
            pickle.dump(
                [dataset.observed_values, dataset.observed_masks, dataset.gt_masks], f
            )

    # Now the path exists, so the dataset object initialization performs data loading.
    ## Difference with 'tabular_Dataset(missing_ratio=missing_ratio, seed=seed, p_obs=p_obs, mecha=mecha, opt=opt)' ?? 
    train_dataset = tabular_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = tabular_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = tabular_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader
