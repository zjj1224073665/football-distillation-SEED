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
import random


class Player():

  def __init__(self):
    self._observation = None
    self._last_action = football_action_set.action_idle
    self._shoot_distance = 0.15
    self._pressure_enabled = False


  def _get_action(self):
    random_number = random.randint(0, 18)
    return random_number

  def take_action(self, observations):
    self._observation = observations
    self._last_action = self._get_action()
    return self._last_action