import argparse
import torch
import datetime
import json
import yaml
import os

from src.main_model_table import TabCSDI
from src.utils_table import train, evaluate_analog, evaluate_analog_all, get_real
from dataset_census.dataset_census_analog2 import get_dataloader

parser = argparse.ArgumentParser(description="TabCSDI")
parser.add_argument("--config", type=str, default="census_onehot_analog.yaml")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--mecha", type=str, default="MAR")
parser.add_argument("--opt", type=str, default="logistic")
parser.add_argument("--p_obs", type=float, default=0.2)
### missingratio, mecha, p_obs 1 추가 (2차(gt_mask)가 기본, 1차는 1 붙은거(observed_masks))
parser.add_argument("--missingratio1", type=float, default=0.2)
parser.add_argument("--mecha1", type=str, default="MAR")
parser.add_argument("--p_obs1", type=float, default=0.2)

args = parser.parse_args()
print(args)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
###
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'
###

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
###
config["model"]["mecha"] = args.mecha
config["model"]["p_obs"] = args.p_obs
###

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# foldername = "./save/census_fold" + str(args.nfold) + "_" + current_time + "/"
foldername = "./save/census_analog_fold" + str(args.nfold) + "/" + args.mecha1 + args.mecha + "_" + current_time + "/" ###
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

full_loader, train_loader, valid_loader = get_dataloader( ### test_loader 뺌
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio1=args.missingratio1,
    mecha1=args.mecha1,
    p_obs1=args.p_obs1,
    missing_ratio2=config["model"]["test_missing_ratio"],
    mecha2=config["model"]["mecha"],
    opt=args.opt,
    p_obs2=config["model"]["p_obs"]
)

model = TabCSDI(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
print("---------------Start testing---------------")
exe_name = "census"
# evaluate_analog(
#     exe_name, model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername
# )
###
evaluate_analog_all(
    exe_name, model, full_loader, nsample=args.nsample, scaler=1, foldername=foldername
)
##
# get_real(model, full_loader, foldername=foldername)