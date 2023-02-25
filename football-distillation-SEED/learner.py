import gfootball.env as football_env
import time, pprint, importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp 
from tensorboardX import SummaryWriter

def write_summary(writer, arg_dict, summary_queue, n_game, loss_lst, pi_loss_lst, v_loss_lst, \
                  entropy_lst, move_entropy_lst, time_steps, self_play_board, win_evaluation, score_evaluation):
    win, score, tot_reward, game_len = [], [], [], []
    loop_t, forward_t, wait_t = [], [], []

    for i in range(arg_dict["summary_game_window"]):
        game_data = summary_queue.get()
        a,b,c,d,opp_num,t1,t2,t3 = game_data
        if arg_dict["env"] == "11_vs_11_kaggle":
            if opp_num in self_play_board:
                self_play_board[opp_num].append(a)
            else:
                self_play_board[opp_num] = [a]

        if 'env_evaluation' in arg_dict and opp_num==arg_dict['env_evaluation']:
            win_evaluation.append(a)
            score_evaluation.append(b)
        else:
            win.append(a)
            score.append(b)
            tot_reward.append(c)
            game_len.append(d)
            loop_t.append(t1)
            forward_t.append(t2)
            wait_t.append(t3)
            
    writer.add_scalar('game/win_rate', float(np.mean(win)), n_game)
    writer.add_scalar('game/score', float(np.mean(score)), n_game)
    writer.add_scalar('game/reward', float(np.mean(tot_reward)), n_game)
    writer.add_scalar('game/game_len', float(np.mean(game_len)), n_game)
    writer.add_scalar('train/step', float(time_steps), n_game)
    writer.add_scalar('time/loop', float(np.mean(loop_t)), n_game)
    writer.add_scalar('time/forward', float(np.mean(forward_t)), n_game)
    writer.add_scalar('time/wait', float(np.mean(wait_t)), n_game)
    writer.add_scalar('train/loss', np.mean(loss_lst), n_game)
    writer.add_scalar('train/pi_loss', np.mean(pi_loss_lst), n_game)
    writer.add_scalar('train/v_loss', np.mean(v_loss_lst), n_game)
    writer.add_scalar('train/entropy', np.mean(entropy_lst), n_game)
    writer.add_scalar('train/move_entropy', np.mean(move_entropy_lst), n_game)

    mini_window = max(1, int(arg_dict['summary_game_window']/3))
    if len(win_evaluation)>=mini_window:
        writer.add_scalar('game/win_rate_evaluation', float(np.mean(win_evaluation)), n_game)
        writer.add_scalar('game/score_evaluation', float(np.mean(score_evaluation)), n_game)
        win_evaluation, score_evaluation = [], []
    
    for opp_num in self_play_board:
        if len(self_play_board[opp_num]) >= mini_window:
            label = 'self_play/'+opp_num
            writer.add_scalar(label, np.mean(self_play_board[opp_num][:mini_window]), n_game)
            self_play_board[opp_num] = self_play_board[opp_num][mini_window:]

    return win_evaluation, score_evaluation

def save_model(model, arg_dict, time_steps, last_saved_step):
    if time_steps >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'time_steps': time_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["log_dir"]+"/model_"+str(time_steps)+".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return time_steps
    else:
        return last_saved_step
        
def get_data(queue, arg_dict, model):
    data = []
    for i in range(arg_dict["buffer_size"]):
        mini_batch_np = []
        for j in range(arg_dict["batch_size"]):
            rollout = queue.get()
            mini_batch_np.append(rollout)
        mini_batch = model.make_batch(mini_batch_np)
        data.append(mini_batch)
    return data

def learner(center_model,sam_model, queue, signal_queue, summary_queue, arg_dict):
    print("Learner process started")
    imported_model = importlib.import_module("models." + arg_dict["model"])
    imported_algo = importlib.import_module("algos." + arg_dict["algorithm"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = imported_model.Model(arg_dict, device)
    model.load_state_dict(center_model.state_dict())
    model.optimizer.load_state_dict(center_model.optimizer.state_dict())
    algo = imported_algo.Algo(arg_dict)
    
    for state in model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    model.to(device)

    for state in sam_model.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    sam_model.to(device)

    time_steps = 0
    if "time_steps" in arg_dict:
        time_steps = arg_dict["time_steps"]
    last_saved_step = time_steps
    
    while True:
        if arg_dict["work_mode"] == "actor":
            if queue.qsize() > arg_dict["batch_size"]*arg_dict["buffer_size"]:
                signal_queue.put(1)
                data = get_data(queue, arg_dict, model)
                torch.save(data, "data/ReplyBuffer.pt")
                break
        elif arg_dict["work_mode"] == "learner":
            last_saved_step = save_model(model, arg_dict, time_steps, last_saved_step)
            signal_queue.put(1)
            data = torch.load("data/ReplyBuffer.pt")
            algo.train(model, sam_model, data, arg_dict, time_steps, last_saved_step)
        else:
            time.sleep(0.1)
            