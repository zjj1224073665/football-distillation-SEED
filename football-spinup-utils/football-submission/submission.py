from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gfootball.env import football_action_set
from gfootball.env import player_base
import numpy as np
import gfootball.env as football_env
import time, pprint, importlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import os
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
from gfootball.env import football_action_set
from gfootball.env import player_base
import numpy as np

class Player():

    def __init__(self, ):

        self._observation = None
        self._last_action = football_action_set.action_idle
        self._shoot_distance = 0.176  # 0.15
        self._pressure_enabled = False
        self._dribble_enabled = False

    def _object_distance(self, object1, object2):
        """Computes distance between two objects."""
        return np.linalg.norm(np.array(object1) - np.array(object2))

    def _direction_action(self, delta):
        """For required movement direction vector returns appropriate action."""
        all_directions = [
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            4
        ]
        all_directions_vec = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1),
                              (1, 0), (1, -1)]
        all_directions_vec = [
            np.array(v) / np.linalg.norm(np.array(v)) for v in all_directions_vec
        ]
        best_direction = np.argmax([np.dot(delta, v) for v in all_directions_vec])
        return all_directions[best_direction]

    def _closest_opponent_to_object(self, o):
        """For a given object returns the closest opponent.
    Args:
      o: Source object.
    Returns:
      Closest opponent."""
        min_d = None
        closest = None
        for p in self._observation['right_team']:
            d = self._object_distance(o, p)
            if min_d is None or d < min_d:
                min_d = d
                closest = p
        assert closest is not None
        return closest

    def _closest_front_opponent(self, o, target):
        """For an object and its movement direction returns the closest opponent.
    Args:
      o: Source object.
      target: Movement direction.
    Returns:
      Closest front opponent."""
        delta = target - o
        min_d = None
        closest = None
        for p in self._observation['right_team']:
            delta_opp = p - o
            if np.dot(delta, delta_opp) <= 0:
                continue
            d = self._object_distance(o, p)
            if min_d is None or d < min_d:
                min_d = d
                closest = p

        # May return None!
        return closest

    def _score_pass_target(self, active, player):
        """Computes score of the pass between players.
    Args:
      active: Player doing the pass.
      player: Player receiving the pass.
    Returns:
      Score of the pass.
    """
        opponent = self._closest_opponent_to_object(player)
        dist = self._object_distance(player, opponent)
        trajectory = player - active
        dist_closest_traj = None
        for i in range(10):
            position = active + (i + 1) / 10.0 * trajectory
            opp_traj = self._closest_opponent_to_object(position)
            dist_traj = self._object_distance(position, opp_traj)
            if dist_closest_traj is None or dist_traj < dist_closest_traj:
                dist_closest_traj = dist_traj
        return -dist_closest_traj

    def _best_pass_target(self, active):
        """Computes best pass a given player can do.
    Args:
      active: Player doing the pass.
    Returns:
      Best target player receiving the pass.
    """
        best_score = None
        best_target = None
        for player in self._observation['left_team']:
            if self._object_distance(player, active) > 0.3:
                continue
            score = self._score_pass_target(active, player)
            if best_score is None or score > best_score:
                best_score = score
                best_target = player
        return best_target

    def _avoid_opponent(self, active, opponent, target):
        """Computes movement action to avoid a given opponent.
    Args:
      active: Active player.
      opponent: Opponent to be avoided.
      target: Original movement direction of the active player.
    Returns:
      Action to perform to avoid the opponent.
    """
        # Choose a perpendicular direction to the opponent, towards the target.
        delta = opponent - active
        delta_t = target - active
        new_delta = [delta[1], -delta[0]]
        if delta_t[0] * new_delta[0] < 0:
            new_delta = [-new_delta[0], -new_delta[1]]

        return self._direction_action(new_delta)

    def _get_action(self):
        """Returns action to perform for the current observations."""
        active = self._observation['left_team'][self._observation['active']]
        # 射门
        if self._observation['ball'][0] >= 0.7 and np.abs(self._observation['ball'][1]) <= 0.264:
            return football_action_set.action_shot
        # 左边
        # 短传
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] <= -0.824 and np.abs(
                self._observation['ball'][1]) >= 0.14:
            return football_action_set.action_short_pass

        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] < 0 and np.abs(
                self._observation['ball'][1]) < 0.132:
            if self._observation['ball'][1] > 0:
                return football_action_set.action_bottom_right
            else:
                return football_action_set.action_top_right
        # 带球向前推进
        if self._observation['ball'][0] <= 0.3 and self._observation['ball_owned_team'] == 0:
            return football_action_set.action_right

        # 右边
        # Corner etc. - just pass the ball
        if self._observation['game_mode'] != 0:
            return football_action_set.action_long_pass
        # 传中
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.9 and np.abs(
                self._observation['ball'][1]) >= 0.35:
            return football_action_set.action_high_pass

        # 外切
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.85 and np.abs(
                self._observation['ball'][1]) >= 0.132:
            if self._observation['ball'][1] > 0:
                return football_action_set.action_top_left
            else:
                return football_action_set.action_bottom_left

        # 上/下传
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.80 and np.abs(
                self._observation['ball'][1]) >= 0.132:
            if self._observation['ball'][1] > 0:
                return football_action_set.action_top
            else:
                return football_action_set.action_bottom

        # 传中
        if self._observation['ball_owned_team'] != 1 and 0.80 > self._observation['ball'][0] > 0.3 and 0.32 >= np.abs(
                self._observation['ball'][1]) >= 0.132:
            if self._observation['ball'][1] > 0:
                return football_action_set.action_bottom_right
            else:
                return football_action_set.action_top_right

        # 带球向前推进
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] < 0.9:
            return football_action_set.action_right

        # 上buff
        if self._observation['ball_owned_team'] != 1 and 0.7 > self._observation['ball'][0] > 0.3 and np.abs(
                self._observation['ball'][1]) < 0.132:
            if self._last_action == football_action_set.action_right:
                self._dribble_enabled = True
                return football_action_set.action_dribble

        if self._dribble_enabled:
            self._dribble_enabled = False
            return football_action_set.action_release_dribble

        if self._observation['ball_owned_team'] == 1:
            if self._last_action == football_action_set.action_pressure:
                return football_action_set.action_sprint
            self._pressure_enabled = True
            return football_action_set.action_pressure

        if self._pressure_enabled:
            self._pressure_enabled = False
            return football_action_set.action_release_pressure
        target_x = 0.824

        if (self._shoot_distance >
                np.linalg.norm(self._observation['ball'][:2] - [target_x, 0])):
            return football_action_set.action_shot

        move_target = [target_x, 0]
        # Compute run direction.
        move_action = self._direction_action(move_target - active)

        closest_front_opponent = self._closest_front_opponent(active, move_target)
        if closest_front_opponent is not None:
            dist_front_opp = self._object_distance(active, closest_front_opponent)
        else:
            dist_front_opp = 2.0

        # Maybe avoid opponent on your way?
        if dist_front_opp < 0.132:
            best_pass_target = self._best_pass_target(active)
            if np.array_equal(best_pass_target, active):
                move_action = self._avoid_opponent(active, closest_front_opponent,
                                                   move_target)
            else:
                delta = best_pass_target - active
                direction_action = self._direction_action(delta)
                if self._last_action == direction_action:
                    return football_action_set.action_short_pass
                else:
                    return direction_action
        return move_action

    def take_action(self, observations):

        self._observation = observations
        self._last_action = self._get_action()
        return self._last_action

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
    # print(a_prob)
    # print(a)
    # categorical(probs)
    # 创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数，K是probs参数的长度
    # 也就是说，按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引
    # 如果probs是长度为K的一维列表，则每个元素是对该索引处的类进行采样的相对概率
    # 例如
    # probs = torch.FloatTensor([0.9,0.2]) #按照9：2的比例采样，概率之和不必为1
    # D = Categorical(probs)
    # for i in range(5): #连续五次采样
    #     print(D.sample()) # tensor(0) tensor(0) tensor(0) tensor(0) tensor(1)
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

class FeatureEncoder:
    def __init__(self):
        self.active = -1
        self.player_pos_x, self.player_pos_y = 0, 0

    def get_feature_dims(self):
        dims = {
            'player': 29,
            'ball': 18,
            'left_team': 7,
            'left_team_closest': 7,
            'right_team': 7,
            'right_team_closest': 7,
        }
        return dims

    def encode(self, obs):
        player_num = obs['active']

        player_pos_x, player_pos_y = obs['left_team'][player_num]
        player_direction = np.array(obs['left_team_direction'][player_num])
        player_speed = np.linalg.norm(player_direction)
        player_role = obs['left_team_roles'][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs['left_team_tired_factor'][player_num]
        is_dribbling = obs['sticky_actions'][9]
        is_sprinting = obs['sticky_actions'][8]

        ball_x, ball_y, ball_z = obs['ball']
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs['ball_direction']
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        if obs['ball_owned_team'] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs['ball_owned_team'] == 0:
            ball_owned_by_us = 1.0
        elif obs['ball_owned_team'] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0

        avail = self._get_avail(obs, ball_distance)
        player_state = np.concatenate(
            (avail[2:], obs['left_team'][player_num], player_direction * 100, [player_speed * 100],
             player_role_onehot, [ball_far, player_tired, is_dribbling, is_sprinting]))

        ball_state = np.concatenate((np.array(obs['ball']),
                                     np.array(ball_which_zone),
                                     np.array([ball_x_relative, ball_y_relative]),
                                     np.array(obs['ball_direction']) * 20,
                                     np.array([ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us])))

        obs_left_team = np.delete(obs['left_team'], player_num, axis=0)
        obs_left_team_direction = np.delete(obs['left_team_direction'], player_num, axis=0)
        left_team_relative = obs_left_team
        left_team_distance = np.linalg.norm(left_team_relative - obs['left_team'][player_num], axis=1, keepdims=True)
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_tired = np.delete(obs['left_team_tired_factor'], player_num, axis=0).reshape(-1, 1)
        left_team_state = np.concatenate((left_team_relative * 2, obs_left_team_direction * 100, left_team_speed * 100, \
                                          left_team_distance * 2, left_team_tired), axis=1)
        left_closest_idx = np.argmin(left_team_distance)
        left_closest_state = left_team_state[left_closest_idx]

        obs_right_team = np.array(obs['right_team'])
        obs_right_team_direction = np.array(obs['right_team_direction'])
        right_team_distance = np.linalg.norm(obs_right_team - obs['left_team'][player_num], axis=1, keepdims=True)
        right_team_speed = np.linalg.norm(obs_right_team_direction, axis=1, keepdims=True)
        right_team_tired = np.array(obs['right_team_tired_factor']).reshape(-1, 1)
        right_team_state = np.concatenate((obs_right_team * 2, obs_right_team_direction * 100, right_team_speed * 100, \
                                           right_team_distance * 2, right_team_tired), axis=1)
        right_closest_idx = np.argmin(right_team_distance)
        right_closest_state = right_team_state[right_closest_idx]

        state_dict = {"player": player_state,
                      "ball": ball_state,
                      "left_team": left_team_state,
                      "left_closest": left_closest_state,
                      "right_team": right_team_state,
                      "right_closest": right_closest_state,
                      "avail": avail}

        return state_dict

    def _get_avail(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        NO_OP, MOVE, LONG_PASS, HIGH_PASS, SHORT_PASS, SHOT, SPRINT, RELEASE_MOVE, \
            RELEASE_SPRINT, SLIDE, DRIBBLE, RELEASE_DRIBBLE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

        if obs['ball_owned_team'] == 1:  # opponents owning ball
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        elif obs['ball_owned_team'] == -1 and ball_distance > 0.03 and obs[
            'game_mode'] == 0:  # Ground ball  and far from me
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS], avail[SHOT], avail[DRIBBLE] = 0, 0, 0, 0, 0
        else:  # my team owning ball
            avail[SLIDE] = 0

        # Dealing with sticky actions
        sticky_actions = obs['sticky_actions']
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs['ball']
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (-0.27 <= ball_y and ball_y <= 0.27):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs['game_mode'] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs['game_mode'] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs['game_mode'] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def _encode_role_onehot(self, role_num):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[role_num] = 1.0
        return np.array(result)

class Model(nn.Module):
    def __init__(self, arg_dict, device=None):
        super(Model, self).__init__()
        self.device = None
        if device:
            self.device = device

        self.arg_dict = arg_dict

        self.fc_player = nn.Linear(arg_dict["feature_dims"]["player"], 64)
        self.fc_ball = nn.Linear(arg_dict["feature_dims"]["ball"], 64)
        self.fc_left = nn.Linear(arg_dict["feature_dims"]["left_team"], 48)
        self.fc_right = nn.Linear(arg_dict["feature_dims"]["right_team"], 48)
        self.fc_left_closest = nn.Linear(arg_dict["feature_dims"]["left_team_closest"], 48)
        self.fc_right_closest = nn.Linear(arg_dict["feature_dims"]["right_team_closest"], 48)

        self.conv1d_left = nn.Conv1d(48, 36, 1, stride=1)
        self.conv1d_right = nn.Conv1d(48, 36, 1, stride=1)
        self.fc_left2 = nn.Linear(36 * 10, 96)
        self.fc_right2 = nn.Linear(36 * 11, 96)
        self.fc_cat = nn.Linear(96 + 96 + 64 + 64 + 48 + 48, arg_dict["lstm_size"])

        self.norm_player = nn.LayerNorm(64)
        self.norm_ball = nn.LayerNorm(64)
        self.norm_left = nn.LayerNorm(48)
        self.norm_left2 = nn.LayerNorm(96)
        self.norm_left_closest = nn.LayerNorm(48)
        self.norm_right = nn.LayerNorm(48)
        self.norm_right2 = nn.LayerNorm(96)
        self.norm_right_closest = nn.LayerNorm(48)
        self.norm_cat = nn.LayerNorm(arg_dict["lstm_size"])

        self.lstm = nn.LSTM(arg_dict["lstm_size"], arg_dict["lstm_size"])

        self.fc_pi_a1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_a2 = nn.Linear(164, 12)
        self.norm_pi_a1 = nn.LayerNorm(164)

        self.fc_pi_m1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.fc_pi_m2 = nn.Linear(164, 8)
        self.norm_pi_m1 = nn.LayerNorm(164)

        self.fc_v1 = nn.Linear(arg_dict["lstm_size"], 164)
        self.norm_v1 = nn.LayerNorm(164)
        self.fc_v2 = nn.Linear(164, 1, bias=False)
        self.optimizer = optim.Adam(self.parameters(), lr=arg_dict["learning_rate"])

    def forward(self, state_dict):
        player_state = state_dict["player"]
        ball_state = state_dict["ball"]
        left_team_state = state_dict["left_team"]
        left_closest_state = state_dict["left_closest"]
        right_team_state = state_dict["right_team"]
        right_closest_state = state_dict["right_closest"]
        avail = state_dict["avail"]
        player_embed = self.norm_player(self.fc_player(player_state))
        ball_embed = self.norm_ball(self.fc_ball(ball_state))
        left_team_embed = self.norm_left(self.fc_left(left_team_state))  # horizon, batch, n, dim
        left_closest_embed = self.norm_left_closest(self.fc_left_closest(left_closest_state))
        right_team_embed = self.norm_right(self.fc_right(right_team_state))
        right_closest_embed = self.norm_right_closest(self.fc_right_closest(right_closest_state))

        [horizon, batch_size, n_player, dim] = left_team_embed.size()
        left_team_embed = left_team_embed.view(horizon * batch_size, n_player, dim).permute(0, 2,
                                                                                            1)  # horizon * batch, dim1, n
        left_team_embed = F.relu(self.conv1d_left(left_team_embed)).permute(0, 2, 1)  # horizon * batch, n, dim2
        left_team_embed = left_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size,
                                                                                 -1)  # horizon, batch, n * dim2
        left_team_embed = F.relu(self.norm_left2(self.fc_left2(left_team_embed)))

        right_team_embed = right_team_embed.view(horizon * batch_size, n_player + 1, dim).permute(0, 2,
                                                                                                  1)  # horizon * batch, dim1, n
        right_team_embed = F.relu(self.conv1d_right(right_team_embed)).permute(0, 2, 1)  # horizon * batch, n * dim2
        right_team_embed = right_team_embed.reshape(horizon * batch_size, -1).view(horizon, batch_size, -1)
        right_team_embed = F.relu(self.norm_right2(self.fc_right2(right_team_embed)))

        cat = torch.cat(
            [player_embed, ball_embed, left_team_embed, right_team_embed, left_closest_embed, right_closest_embed], 2)
        cat = F.relu(self.norm_cat(self.fc_cat(cat)))
        h_in = state_dict["hidden"]
        out, h_out = self.lstm(cat, h_in)

        a_out = F.relu(self.norm_pi_a1(self.fc_pi_a1(out)))
        a_out = self.fc_pi_a2(a_out)
        logit = a_out + (avail - 1) * 1e7
        prob = F.softmax(logit, dim=2)

        prob_m = F.relu(self.norm_pi_m1(self.fc_pi_m1(out)))
        prob_m = self.fc_pi_m2(prob_m)
        prob_m = F.softmax(prob_m, dim=2)

        v = F.relu(self.norm_v1(self.fc_v1(out)))
        v = self.fc_v2(v)

        return prob, prob_m, v, h_out

    def make_batch(self, data):
        # data = [tr1, tr2, ..., tr10] * batch_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        s_player_batch, s_ball_batch, s_left_batch, s_left_closest_batch, s_right_batch, s_right_closest_batch, avail_batch = [], [], [], [], [], [], []
        s_player_prime_batch, s_ball_prime_batch, s_left_prime_batch, s_left_closest_prime_batch, \
            s_right_prime_batch, s_right_closest_prime_batch, avail_prime_batch = [], [], [], [], [], [], []
        h1_in_batch, h2_in_batch, h1_out_batch, h2_out_batch = [], [], [], []
        a_batch, m_batch, r_batch, prob_batch, done_batch, need_move_batch = [], [], [], [], [], []

        for rollout in data:
            s_player_lst, s_ball_lst, s_left_lst, s_left_closest_lst, s_right_lst, s_right_closest_lst, avail_lst = [], [], [], [], [], [], []
            s_player_prime_lst, s_ball_prime_lst, s_left_prime_lst, s_left_closest_prime_lst, \
                s_right_prime_lst, s_right_closest_prime_lst, avail_prime_lst = [], [], [], [], [], [], []
            h1_in_lst, h2_in_lst, h1_out_lst, h2_out_lst = [], [], [], []
            a_lst, m_lst, r_lst, prob_lst, done_lst, need_move_lst = [], [], [], [], [], []

            for transition in rollout:
                s, a, m, r, s_prime, prob, done, need_move = transition

                s_player_lst.append(s["player"])
                s_ball_lst.append(s["ball"])
                s_left_lst.append(s["left_team"])
                s_left_closest_lst.append(s["left_closest"])
                s_right_lst.append(s["right_team"])
                s_right_closest_lst.append(s["right_closest"])
                avail_lst.append(s["avail"])
                h1_in, h2_in = s["hidden"]
                h1_in_lst.append(h1_in)
                h2_in_lst.append(h2_in)

                s_player_prime_lst.append(s_prime["player"])
                s_ball_prime_lst.append(s_prime["ball"])
                s_left_prime_lst.append(s_prime["left_team"])
                s_left_closest_prime_lst.append(s_prime["left_closest"])
                s_right_prime_lst.append(s_prime["right_team"])
                s_right_closest_prime_lst.append(s_prime["right_closest"])
                avail_prime_lst.append(s_prime["avail"])
                h1_out, h2_out = s_prime["hidden"]
                h1_out_lst.append(h1_out)
                h2_out_lst.append(h2_out)

                a_lst.append([a])
                m_lst.append([m])
                r_lst.append([r])
                prob_lst.append([prob])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                need_move_lst.append([need_move]),

            s_player_batch.append(s_player_lst)
            s_ball_batch.append(s_ball_lst)
            s_left_batch.append(s_left_lst)
            s_left_closest_batch.append(s_left_closest_lst)
            s_right_batch.append(s_right_lst)
            s_right_closest_batch.append(s_right_closest_lst)
            avail_batch.append(avail_lst)
            h1_in_batch.append(h1_in_lst[0])
            h2_in_batch.append(h2_in_lst[0])

            s_player_prime_batch.append(s_player_prime_lst)
            s_ball_prime_batch.append(s_ball_prime_lst)
            s_left_prime_batch.append(s_left_prime_lst)
            s_left_closest_prime_batch.append(s_left_closest_prime_lst)
            s_right_prime_batch.append(s_right_prime_lst)
            s_right_closest_prime_batch.append(s_right_closest_prime_lst)
            avail_prime_batch.append(avail_prime_lst)
            h1_out_batch.append(h1_out_lst[0])
            h2_out_batch.append(h2_out_lst[0])

            a_batch.append(a_lst)
            m_batch.append(m_lst)
            r_batch.append(r_lst)
            prob_batch.append(prob_lst)
            done_batch.append(done_lst)
            need_move_batch.append(need_move_lst)

        s = {
            "player": torch.tensor(s_player_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_batch, dtype=torch.float, device=device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "right_team": torch.tensor(s_right_batch, dtype=torch.float, device=device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "avail": torch.tensor(avail_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_in_batch, dtype=torch.float, device=device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_in_batch, dtype=torch.float, device=device).squeeze(1).permute(1, 0, 2))
        }

        s_prime = {
            "player": torch.tensor(s_player_prime_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "ball": torch.tensor(s_ball_prime_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "left_team": torch.tensor(s_left_prime_batch, dtype=torch.float, device=device).permute(1, 0, 2, 3),
            "left_closest": torch.tensor(s_left_closest_prime_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "right_team": torch.tensor(s_right_prime_batch, dtype=torch.float, device=device).permute(1, 0, 2, 3),
            "right_closest": torch.tensor(s_right_closest_prime_batch, dtype=torch.float, device=device).permute(1, 0,
                                                                                                                 2),
            "avail": torch.tensor(avail_prime_batch, dtype=torch.float, device=device).permute(1, 0, 2),
            "hidden": (torch.tensor(h1_out_batch, dtype=torch.float, device=device).squeeze(1).permute(1, 0, 2),
                       torch.tensor(h2_out_batch, dtype=torch.float, device=device).squeeze(1).permute(1, 0, 2))
        }

        a, m, r, done_mask, prob, need_move = torch.tensor(a_batch, device=device).permute(1, 0, 2), \
            torch.tensor(m_batch, device=device).permute(1, 0, 2), \
            torch.tensor(r_batch, dtype=torch.float, device=device).permute(1, 0, 2), \
            torch.tensor(done_batch, dtype=torch.float, device=device).permute(1, 0, 2), \
            torch.tensor(prob_batch, dtype=torch.float, device=device).permute(1, 0, 2), \
            torch.tensor(need_move_batch, dtype=torch.float, device=device).permute(1, 0, 2)

        return s, a, m, r, s_prime, done_mask, prob, need_move

def my_controller(observation, action_space, is_act_continuous=False):
    arg_dict = {
        "env": "11_vs_11_kaggle",
        # "11_vs_11_kaggle" : environment used for self-play training
        # "11_vs_11_stochastic" : environment used for training against fixed opponent(rule-based AI)
        "lstm_size": 256,
        "learning_rate": 0.0001,
        "encoder": "encoder_basic",
        "rewarder": "rewarder_basic",
        "model": "conv1d",
        "heatmap": False,
        "rule_based": False
    }
    fe = FeatureEncoder()
    obs = [observation]
    arg_dict["feature_dims"] = fe.get_feature_dims()
    action = 0
    if arg_dict["rule_based"] == False:
        initial_flag = 0
        if initial_flag == 0:
            h_out = (torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float),
                     torch.zeros([1, 1, arg_dict["lstm_size"]], dtype=torch.float))
        h_in = h_out
        state_dict = fe.encode(obs[0])
        state_dict_tensor = state_to_tensor(state_dict, h_in)
        cen_model = Model(arg_dict)
        current_path = os.path.abspath(__file__)
        parent_path = os.path.dirname(current_path)

        cen_path = parent_path + '/' + model + '.pt'
        cpu_device = torch.device('cpu')
        cen_checkpoint = torch.load(cen_path, map_location=cpu_device)
        cen_model.load_state_dict(cen_checkpoint)
        with torch.no_grad():
            a_prob, m_prob, _, h_out = cen_model(state_dict_tensor)  # 计算a,m,h
            # torch.no_grad()可以让节点不进行求梯度，从而节省了内存控件，当神经网络较大且内存不够用时，就需要让梯度为False
        real_action, _, _, _, _, _, _ = get_action(a_prob, m_prob)  # 更新动作
        action = real_action
        initial_flag += 1
    else:
        player = Player()
        action = player.take_action(obs[0])
    action_final = [[0] * 19]
    action_final[0][action] = 1
    return action_final

model = 'model_13mil'

if __name__ == '__main__':

    checkpoint = torch.load(model + '.tar')
    torch.save(checkpoint['model_state_dict'], model + '.pt')
    # torch.set_printoptions(threshold=np.inf)
    # data = open("model.txt", 'w', encoding="utf-8")
    # print(checkpoint['model_state_dict'], file=data)

    # current_path = os.path.abspath(__file__)
    # parent_path = os.path.dirname(current_path)
    # print(parent_path+ '/model.pt')





