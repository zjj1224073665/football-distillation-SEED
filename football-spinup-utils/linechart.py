import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import gfootball.env as football_env
import time, pprint, importlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pickle
from scipy.interpolate import make_interp_spline

def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

def state_to_tensor(state_dict, h_in):
    # torch.from_numpy() 把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
    # .unsqueeze(i) 增加第i个维度
    player_state = torch.from_numpy(state_dict["player"]).float().unsqueeze(0).unsqueeze(0)
    ball_state = torch.from_numpy(state_dict["ball"]).float().unsqueeze(0).unsqueeze(0)
    left_team_state = torch.from_numpy(state_dict["left_team"]).float().unsqueeze(0).unsqueeze(0)
    left_closest_state = torch.from_numpy(state_dict["left_closest"]).float().unsqueeze(0).unsqueeze(0)
    right_team_state = torch.from_numpy(state_dict["right_team"]).float().unsqueeze(0).unsqueeze(0)
    right_closest_state = torch.from_numpy(state_dict["right_closest"]).float().unsqueeze(0).unsqueeze(0)
    avail = torch.from_numpy(state_dict["avail"]).float().unsqueeze(0).unsqueeze(0)

    state_dict_tensor = {
        "player": player_state,
        "ball": ball_state,
        "left_team": left_team_state,
        "left_closest": left_closest_state,
        "right_team": right_team_state,
        "right_closest": right_closest_state,
        "avail": avail,
        "hidden": h_in  # 两个1*1*256的张量合集 LSTM_size =256 采用LSTM替换DQN卷基层后的一个全连接层，来达到能够记忆历史状态的作用
    }  # 定义状态
    return state_dict_tensor


def get_action(a_prob, m_prob):
    a = Categorical(a_prob).sample().item()
    m, need_m = 0, 0
    prob_selected_a = a_prob[0][0][a].item()
    prob_selected_m = 0
    if a == 0:  # 预定动作
        real_action = a
        prob = prob_selected_a
    elif a == 1:  # 8个动作集合
        m = Categorical(m_prob).sample().item()  # 从8个动作中选择一个
        need_m = 1
        real_action = m + 1  # 真实的动作
        prob_selected_m = m_prob[0][0][m].item()
        prob = prob_selected_a * prob_selected_m
    else:
        real_action = a + 7  # 从其他动作中选择
        prob = prob_selected_a

    assert prob != 0, 'prob 0 ERROR!!!! a : {}, m:{}  {}, {}'.format(a, m, prob_selected_a, prob_selected_m)
    # 概率为0 ，说明没有选择动作，弹出警告
    return real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m

def summary_data_writer(arg_dict):
    evaluate_path = arg_dict["evaluate_path"][2]
    save_path = arg_dict["summary_data_path"][2]
    print("Evaluator process started")
    print("evaluate_path:{}".format(evaluate_path)+"\n" + "save_path:{}".format(save_path))
    mean_score = []
    std_score = []
    mean_win = []
    std_win = []
    imported_model = importlib.import_module("football-bot.models." + arg_dict["model"])
    fe_module = importlib.import_module("football-bot.encoders." + arg_dict["encoder"])  # 给状态特征编码
    fe = fe_module.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()
    for seed in range(3):
        model_score = []
        std_model_score = []
        model_win = []
        std_model_win = []
        for model_item in [i/2 for i in range(27)]:
            cen_model = imported_model.Model(arg_dict)
            cen_path = evaluate_path + 'model_{}mil.tar'.format(int(model_item + model_item % 1))
            cen_checkpoint = torch.load(cen_path)
            cen_model.load_state_dict(cen_checkpoint['model_state_dict'])
            env = football_env.create_environment(env_name=arg_dict["env"], representation="raw", stacked=False,
                                                  logdir='/tmp/football', \
                                                  write_goal_dumps=False, write_full_episode_dumps=False, render=False)
            roll_score = []
            roll_win = []
            for roll in range(arg_dict["rollouts"]):  # episode loop
                print('seed_{}_model_{}mil_evaluate_process: {}%'.format(seed,model_item,int((roll+1)/arg_dict["rollouts"]*100)))
                env.reset()
                done = False
                steps, score, tot_reward, win = 0, 0, 0, 0
                h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float),
                         torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))

                obs = env.observation()
                while not done:  # step loop
                    h_in = h_out
                    state_dict = fe.encode(obs[0])
                    state_dict_tensor = state_to_tensor(state_dict, h_in)

                    with torch.no_grad():
                        a_prob, m_prob, _, h_out = cen_model(state_dict_tensor)
                    real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

                    prev_obs = obs
                    obs, rew, done, info = env.step(real_action)
                    state_prime_dict = fe.encode(obs[0])

                    (h1_in, h2_in) = h_in
                    (h1_out, h2_out) = h_out
                    state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                    state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                    steps += 1
                    score += rew
                    if done:
                        if score > 0:
                            win = 1
                roll_score.append(score)
                roll_win.append(win)
            model_score.append(np.mean(roll_score))
            std_model_score.append(np.std(roll_score))
            model_win.append(np.mean(roll_win))
            std_model_win.append(np.std(roll_win))
        mean_score.append(model_score)
        std_score.append(std_model_score)
        mean_win.append(model_win)
        std_win.append(std_model_win)
    summary_data = {"mean_score": mean_score, "std_score": std_score, "mean_win": mean_win,
                        "std_win": std_win}
    with open(save_path, "wb") as f:
        pickle.dump(summary_data, f, pickle.HIGHEST_PROTOCOL)

def summary_data_writer_(arg_dict):
    evaluate_path = arg_dict["evaluate_path"][3]
    save_path = arg_dict["summary_data_path"][3]
    print("Evaluator process started")
    print("evaluate_path:{}".format(evaluate_path)+"\n" + "save_path:{}".format(save_path))
    mean_score = []
    std_score = []
    mean_win = []
    std_win = []
    imported_model = importlib.import_module("football-bot.models." + arg_dict["model"])
    fe_module = importlib.import_module("football-bot.encoders." + arg_dict["encoder"])  # 给状态特征编码
    fe = fe_module.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()
    for seed in range(3):
        model_score = []
        std_model_score = []
        model_win = []
        std_model_win = []
        for model_item in [i/2 for i in range(27)]:
            cen_model = imported_model.Model(arg_dict)
            cen_path = evaluate_path + 'model_{}mil.tar'.format(int(model_item + model_item % 1))
            cen_checkpoint = torch.load(cen_path)
            cen_model.load_state_dict(cen_checkpoint['model_state_dict'])
            env = football_env.create_environment(env_name=arg_dict["env"], representation="raw", stacked=False,
                                                  logdir='/tmp/football', \
                                                  write_goal_dumps=False, write_full_episode_dumps=False, render=False)
            roll_score = []
            roll_win = []
            for roll in range(arg_dict["rollouts"]):  # episode loop
                print('seed_{}_model_{}mil_evaluate_process: {}%'.format(seed,model_item,int((roll+1)/arg_dict["rollouts"]*100)))
                env.reset()
                done = False
                steps, score, tot_reward, win = 0, 0, 0, 0
                h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float),
                         torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))

                obs = env.observation()
                while not done:  # step loop
                    h_in = h_out
                    state_dict = fe.encode(obs[0])
                    state_dict_tensor = state_to_tensor(state_dict, h_in)

                    with torch.no_grad():
                        a_prob, m_prob, _, h_out = cen_model(state_dict_tensor)
                    real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

                    prev_obs = obs
                    obs, rew, done, info = env.step(real_action)
                    state_prime_dict = fe.encode(obs[0])

                    (h1_in, h2_in) = h_in
                    (h1_out, h2_out) = h_out
                    state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
                    state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
                    steps += 1
                    score += rew
                    if done:
                        if score > 0:
                            win = 1
                roll_score.append(score)
                roll_win.append(win)
            model_score.append(np.mean(roll_score))
            std_model_score.append(np.std(roll_score))
            model_win.append(np.mean(roll_win))
            std_model_win.append(np.std(roll_win))
        mean_score.append(model_score)
        std_score.append(std_model_score)
        mean_win.append(model_win)
        std_win.append(std_model_win)
    summary_data = {"mean_score": mean_score, "std_score": std_score, "mean_win": mean_win,
                        "std_win": std_win}
    with open(save_path, "wb") as f:
        pickle.dump(summary_data, f, pickle.HIGHEST_PROTOCOL)

def single_model_evaluate(arg_dict):
    print("Evaluator process started")
    print("single_evaluate_path: {}".format(arg_dict["single_evaluate_path"]))
    imported_model = importlib.import_module("football-bot.models." + arg_dict["model"])
    fe_module = importlib.import_module("football-bot.encoders." + arg_dict["encoder"])  # 给状态特征编码
    fe = fe_module.FeatureEncoder()
    arg_dict["feature_dims"] = fe.get_feature_dims()
    cen_model = imported_model.Model(arg_dict)
    cen_path = arg_dict["single_evaluate_path"]
    cen_checkpoint = torch.load(cen_path)
    cen_model.load_state_dict(cen_checkpoint['model_state_dict'])
    env = football_env.create_environment(env_name=arg_dict["env"], representation="raw", stacked=False,
                                          logdir='/tmp/football', \
                                          write_goal_dumps=False, write_full_episode_dumps=False, render=False)
    roll_score = []
    roll_win = []
    for roll in range(arg_dict["rollouts"]):  # episode loop
        print('model_evaluate_process: {}%'.format(int((roll+1)/arg_dict["rollouts"]*100)))
        env.reset()
        done = False
        steps, score, tot_reward, win = 0, 0, 0, 0
        h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float),
                 torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))

        obs = env.observation()
        while not done:  # step loop
            h_in = h_out
            state_dict = fe.encode(obs[0])
            state_dict_tensor = state_to_tensor(state_dict, h_in)

            with torch.no_grad():
                a_prob, m_prob, _, h_out = cen_model(state_dict_tensor)
            real_action, a, m, need_m, prob, prob_selected_a, prob_selected_m = get_action(a_prob, m_prob)

            prev_obs = obs
            obs, rew, done, info = env.step(real_action)
            state_prime_dict = fe.encode(obs[0])

            (h1_in, h2_in) = h_in
            (h1_out, h2_out) = h_out
            state_dict["hidden"] = (h1_in.numpy(), h2_in.numpy())
            state_prime_dict["hidden"] = (h1_out.numpy(), h2_out.numpy())
            steps += 1
            score += rew
            if done:
                if score > 0:
                    win = 1
        roll_score.append(score)
        roll_win.append(win)
    print("win_rate: {}".format(np.mean(roll_win)))
    print("score: {}".format(np.mean(roll_score)))

def win_rate_plot(axs):
    time = np.array([i / 2 for i in range(27)])
    time_smooth = np.linspace(time.min(), time.max(), 300)
    data = []
    data_smooth = []

    for file_name in arg_dict["summary_data_path"]:
        file = file_name
        with open(file, "rb") as f:
            data_temp = pickle.load(f)
        data.append(data_temp["mean_win"])
        data_temp_smooth = []
        # for i in range(len(data_temp["mean_win"])):
            # data_temp_smooth.append(make_interp_spline(time, data_temp["mean_win"][i])(time_smooth))
        data_temp_smooth = smooth(data_temp["mean_win"], sm=3)
        data_smooth.append(data_temp_smooth)
    color = ['r', 'b', 'g', 'm']
    label = ['Counter-Strategies Distillation', 'Self Play', 'League Learning', 'Versus Built-in']
    for i in range(4):
        sns.tsplot(time=time, data=data_smooth[i], color=color[i],ax = axs[0])
    sns.despine()
    axs[0].set_ylabel("Test Win %")
    axs[0].set_xlabel("T(mil)")
    axs[0].set_title("GRF_11_vs_11_medium_stochastic",fontsize = 10)


def score_plot(axs):
    time = np.array([i / 2 for i in range(27)])
    time_smooth = np.linspace(time.min(), time.max(), 300)
    data = []
    data_smooth = []

    for file_name in arg_dict["summary_data_path"]:
        file = file_name
        with open(file, "rb") as f:
            data_temp = pickle.load(f)
        data.append(data_temp["mean_score"])
        data_temp_smooth = []
        # for i in range(len(data_temp["mean_win"])):
            # data_temp_smooth.append(make_interp_spline(time, data_temp["mean_win"][i])(time_smooth))
        data_temp_smooth = smooth(data_temp["mean_score"], sm=3)
        data_smooth.append(data_temp_smooth)
    color = ['r', 'b', 'g', 'm']
    label = ['Counter-Strategies Distillation', 'Self Play', 'League Learning', 'Versus Built-in']
    for i in range(4):
        sns.tsplot(time=time, data=data_smooth[i], color=color[i], condition=label[i])
    sns.despine()
    axs[1].set_ylabel("Reward")
    axs[1].set_xlabel("T(mil)")
    axs[1].set_title("GRF_11_vs_11_medium_stochastic",fontsize = 10)


if __name__ == '__main__':
    arg_dict = {
        "env": "11_vs_11_stochastic",
        "lstm_size": 256,
        "learning_rate": 0.0001,
        "encoder": "encoder_basic",
        "rewarder": "rewarder_basic",
        "model": "conv1d",
        "heatmap": False,
        "single_evaluate_path": None,
        "evaluate_path" : ['football-bot/Counter_Strategies/Main_Agent/[01-25]07.17.45/',"football-bot/Self_Play/","football-bot/League_Learning/[01-31]19.32.50/","football-bot/Versus_Built-in/[02-03]08.05.51/"],
        "summary_data_path": ["summary_data_Counter_Strategies_Distillation.pkl",'summary_data_Self_Play.pkl', 'summary_data_League_Learning.pkl', 'summary_data_Versus_Built-in.pkl'],
        "rollouts" : 10
    }


    # summary_data_writer(arg_dict)
    # summary_data_writer_(arg_dict)
    # single_model_evaluate(arg_dict)
    #
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    win_rate_plot(axs)
    score_plot(axs)
    plt.legend(bbox_to_anchor=(0.8, 1.19), frameon=False, ncol=4,fontsize = 10.5)
    plt.savefig('Figure.pdf')
    plt.show()

    # PTL_plot()

    # Elo_plot_single()
