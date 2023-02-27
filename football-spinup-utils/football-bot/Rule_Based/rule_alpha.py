# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Sample bot player."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gfootball.env import player_base
import numpy as np


class Player():

  def __init__(self,):

    self._observation = None
    self._last_action = 0
    self._shoot_distance = 0.15
    self._pressure_enabled = False

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
    if self._observation['ball'][0] >= 0.7 and self._observation['ball'][1] <= 0.12:
      return 12
    # 带球向前推进
    if self._observation['ball'][0] <= 0.7 and self._observation['ball_owned_team'] == 0:
      return 5
    # 内切
    if self._observation['ball_owned_team'] != 1 and 0.9 > self._observation['ball'][0] > 0.4 and np.abs(self._observation['ball'][1]) >= 0.08:
      if self._observation['ball'][1] > 0:
        return 4
      else:
        return 6
    # 传中
    if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.9 and np.abs(
            self._observation['ball'][1]) >= 0.35:
        return 10
    # 内切
    if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.85 and np.abs(self._observation['ball'][1]) >= 0.08:
      if self._observation['ball'][1] > 0:
        return 2
      else:
        return 8
    # 传球
    if self._observation['ball_owned_team'] == 0:
        press_flag = 0
        for item_ltx in range(1, 5):
            if np.abs(self._observation['right_team'][item_ltx][0] -
                         self._observation['left_team'][self._observation['ball_owned_player']][0]) <= 2e-02 and np.abs(
                self._observation['right_team'][item_ltx][1] - self._observation['left_team'][self._observation['ball_owned_player']][1]) <= 2e-02:
                press_flag = 1
        if press_flag == 1:
            if self._observation['ball'][0] < -0.2:
                return 9
    # 防守
    if self._observation['ball_owned_team'] == 1:
        ball_relative_coordinate = []
        ball_relative_coordinate.append(self._observation['ball'][0]-self._observation['left_team'][self._observation["designated"]][0])
        ball_relative_coordinate.append(self._observation['ball'][1]-self._observation['left_team'][self._observation["designated"]][1])
        if all(i >= 0 for i in ball_relative_coordinate):
            if 2 > np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) > 0.5:
                return 6
            elif np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) <= 0.5:
                return 7
            elif np.abs(ball_relative_coordinate[0] / ball_relative_coordinate[1]) >= 2:
                return 5
            # print('player',self._observation['left_team'][player_id + 1])
            # print('youxia')

        elif all(i <= 0 for i in ball_relative_coordinate):
            if 2 > np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) > 0.5:
                return 2
            elif np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) <= 0.5:
                return 3
            elif np.abs(ball_relative_coordinate[0] / ball_relative_coordinate[1]) >= 2:
                return 1
            # print('player', self._observation['left_team'][player_id + 1])
            # print('zuoshang')
        elif ball_relative_coordinate[0] > 0 and ball_relative_coordinate[1] < 0 :
            if 2 > np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) > 0.5:
                return 4
            elif np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) <= 0.5:
                return 3
            elif np.abs(ball_relative_coordinate[0] / ball_relative_coordinate[1]) >= 2:
              return 5
            # print('player', self._observation['left_team'][player_id + 1])
            # print('youshang')
        elif ball_relative_coordinate[0] < 0 and ball_relative_coordinate[1] > 0:
            if 2>np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) > 0.5:
                return 8
            elif np.abs(ball_relative_coordinate[0]/ball_relative_coordinate[1]) <= 0.5:
              return 7
            elif np.abs(ball_relative_coordinate[0] / ball_relative_coordinate[1]) >= 2:
              return 1
            # print('player', self._observation['left_team'][player_id + 1])
            # print('zuoxia')
        if np.random.randint(0, 10) > 6:
          return 13

    # Corner etc. - just pass the ball
    if self._observation['game_mode'] != 0:
      return 9

    target_x = 0.85


    move_target = [target_x, 0]
    # Compute run direction.
    move_action = self._direction_action(move_target - active)

    closest_front_opponent = self._closest_front_opponent(active, move_target)
    if closest_front_opponent is not None:
      dist_front_opp = self._object_distance(active, closest_front_opponent)
    else:
      dist_front_opp = 2.0

    # Maybe avoid opponent on your way?
    if dist_front_opp < 0.08:
      best_pass_target = self._best_pass_target(active)
      if np.array_equal(best_pass_target, active):
        move_action = self._avoid_opponent(active, closest_front_opponent,
                                          move_target)
      else:
        delta = best_pass_target - active
        direction_action = self._direction_action(delta)
        if self._last_action == direction_action:
          return 11
        else:
          return direction_action
    return move_action


  def take_action(self, observations):

    self._observation = observations
    self._last_action = self._get_action()
    return self._last_action