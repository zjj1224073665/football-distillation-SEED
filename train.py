import gfootball.env as football_env
import time, pprint, json, os, importlib, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from actor import *
from learner import *
from evaluator import evaluator
from datetime import datetime, timedelta


def save_args(arg_dict):
    os.makedirs(arg_dict["log_dir"])
    args_info = json.dumps(arg_dict, indent=4)
    f = open(arg_dict["log_dir"] + "/args.json", "w")
    f.write(args_info)
    f.close()


def copy_models(dir_src, dir_dst):  # src: source, dst: destination
    # retireve list of models
    l_cands = [f for f in os.listdir(dir_src) if os.path.isfile(os.path.join(dir_src, f)) and 'model_' in f]
    l_cands = sorted(l_cands, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"models to be copied: {l_cands}")
    for m in l_cands:
        shutil.copyfile(os.path.join(dir_src, m), os.path.join(dir_dst, m))
    print(f"{len(l_cands)} models copied in the given directory")


def main(arg_dict):
    os.environ['OPENBLAS_NUM_THREADS'] = '6'
    cur_time = datetime.now()
    arg_dict["log_dir"] = "logs/" + cur_time.strftime("[%m-%d]%H.%M.%S")
    save_args(arg_dict)
    if arg_dict["trained_model_path"] and 'kaggle' in arg_dict['env']:
        copy_models(os.path.dirname(arg_dict['trained_model_path']), arg_dict['log_dir'])

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    pp = pprint.PrettyPrinter(indent=4)
    torch.set_num_threads(1)

    fe = importlib.import_module("encoders." + arg_dict["encoder"])
    fe = fe.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()

    model = importlib.import_module("models." + arg_dict["model"])
    cpu_device = torch.device('cpu')
    center_model = model.Model(arg_dict)

    sam_model = model.Model(arg_dict)
    sam_path = arg_dict["sam_path"]
    sam_checkpoint = torch.load(sam_path)
    sam_model.load_state_dict(sam_checkpoint['model_state_dict'])


    if arg_dict["trained_model_path"]:
        checkpoint = torch.load(arg_dict["trained_model_path"], map_location=cpu_device)
        time_steps = checkpoint['optimization_step']
        center_model.load_state_dict(checkpoint['model_state_dict'])
        center_model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        arg_dict["time_steps"] = time_steps
        print("Trained model", arg_dict["trained_model_path"], "suffessfully loaded")
    else:
        time_steps = 0

    model_dict = {
        'time_steps': time_steps,
        'model_state_dict': center_model.state_dict(),
        'optimizer_state_dict': center_model.optimizer.state_dict(),
    }

    path = arg_dict["log_dir"] + f"/model_{time_steps}.tar"
    torch.save(model_dict, path)

    center_model.share_memory()
    data_queue = mp.Queue()
    signal_queue = mp.Queue()
    summary_queue = mp.Queue()

    processes = []
    p = mp.Process(target=learner, args=(center_model,sam_model, data_queue, signal_queue, summary_queue, arg_dict))
    p.start()
    processes.append(p)

    if ["work_mode"]=="actor":
        for rank in range(arg_dict["num_processes"]):
            p = mp.Process(target=actor, args=(rank, sam_model, data_queue, signal_queue, summary_queue, arg_dict))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    arg_dict = {
        "env":"11_vs_11_stochastic",
        "num_processes": 1,
        "batch_size": 32,
        "buffer_size": 2000,
        "rollout_len": 30,

        "lstm_size": 256,
        "k_epoch": 3,
        "learning_rate": 0.0001,
        "gamma": 0.993,
        "lmbda": 0.96,
        "entropy_coef": 0.0001,
        "grad_clip": 3.0,
        "eps_clip": 0.1,

        "summary_game_window": 10,
        "model_save_interval": 150000,

        "trained_model_path":  None,
        "print_mode": False,

        "encoder": "encoder_basic",
        "rewarder": "rewarder_basic",
        "model": "conv1d",
        "algorithm": "distillation",
        "work_mode": "learner",
        "sam_path": "bot/model_sam.tar",

    }

    main(arg_dict)
