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

from gfootball.env import football_action_set
from gfootball.env import player_base
import numpy as np


class Player():

    def __init__(self, ):

        self._observation = None
        self._last_action = 0
        self._shoot_distance = 0.15  # 0.15
        self._sprint_enabled = False

    def _object_distance(self, object1, object2):
        """Computes distance between two objects."""
        return np.linalg.norm(np.array(object1) - np.array(object2))

    def _direction_action(self, delta):
        """For required movement direction vector returns appropriate action."""
        all_directions = [3, 2, 1, 8, 7, 6, 5, 4]
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
        if self._observation['ball'][0] >= 0.70 and np.abs(self._observation['ball'][1]) <= 0.12:
            return 12
        # 左边
        # 短传
        if self._observation['ball_owned_team'] == 0 and self._observation['ball'][0] <= -0.85 and np.abs(
                self._observation['ball'][1]) >= 0.10:
            return 11
        # 外切
        if self._observation['ball_owned_team'] == 0 and self._observation['ball'][0] < 0 and 0.32 >= np.abs(
                self._observation['ball'][1]) >= 0.10:
            if self._observation['ball'][1] > 0:
                return 6
            else:
                return 4
        # 带球向前推进
        if self._observation['ball'][0] <= 0.7 and self._observation['ball_owned_team'] == 0:
            return 5

        # 右边
        # Corner etc. - just pass the ball
        if self._observation['game_mode'] != 0:
            return 9
        # 长传
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.9 and np.abs(
                self._observation['ball'][1]) >= 0.35:
            return 10

        # 短传
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.85 and np.abs(
                self._observation['ball'][1]) >= 0.10:
            return 11

        # 内切
        if self._observation['ball_owned_team'] != 1 and self._observation['ball'][0] >= 0.70 and np.abs(
                self._observation['ball'][1]) >= 0.10:
            if self._observation['ball'][1] > 0:
                return 4
            else:
                return 6

        # 外切
        if self._observation['ball_owned_team'] != 1 and 0.80 > self._observation['ball'][0] > 0.3 and 0.32 >= np.abs(
                self._observation['ball'][1]) >= 0.10:
            if self._observation['ball'][1] > 0:
                return 6
            else:
                return 4

        # 防守
        if self._observation['ball_owned_team'] != 0:
            designated = self._observation['left_team'][self._observation["designated"]]
            ball = self._observation['ball']
            ball_relative_coordinate = [ball[0] - designated[0], ball[1] - designated[1]]
            move_action_2 = self._direction_action(ball_relative_coordinate)
            if self._observation['ball_owned_team'] == 1:
                closest_front_opponent_2 = self._closest_front_opponent(designated[:2], ball[:2])
                if closest_front_opponent_2 is not None:
                    dist_front_opp_2 = self._object_distance(designated, closest_front_opponent_2)
                else:
                    dist_front_opp_2 = 2.17
                if dist_front_opp_2 < 0.05:
                    best_pass_target_2 = self._best_pass_target(designated)
                    if np.array_equal(best_pass_target_2, designated):
                        move_action_2 = self._avoid_opponent(designated, closest_front_opponent_2,
                                                             ball[:2])
                    else:
                        delta_2 = best_pass_target_2 - designated
                        direction_action_2 = self._direction_action(delta_2)
                        if self._last_action == direction_action_2:
                            return 11
                        else:
                            return direction_action_2
                return move_action_2
            elif self._observation['ball_owned_team'] != 1:
                if self._last_action == 13:
                    if self._observation['ball'][0] <= -0.85 and np.abs(self._observation['ball'][1]) >= 0.10:
                        return 11

                    if self._observation['ball'][0] < 0 and 0.32 >= np.abs(self._observation['ball'][1]) >= 0.10:
                        if self._observation['ball'][1] > 0:
                            return 6
                        else:
                            return 4

                self._sprint_enabled = True
                return 13

        if self._sprint_enabled:
            self._sprint_enabled = False
            return 15

        target_x = 0.85
        move_target = [target_x, 0]
        # Compute run direction.
        move_action = self._direction_action(move_target - active)

        closest_front_opponent = self._closest_front_opponent(active, move_target)
        if closest_front_opponent is not None:
            dist_front_opp = self._object_distance(active, closest_front_opponent)
        else:
            dist_front_opp = 2.17

        # Maybe avoid opponent on your way?
        if dist_front_opp < 0.10:
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
