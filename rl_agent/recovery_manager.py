from environment.pick_place_wrapper import PickPlaceWrapper
from enum import Enum

import numpy as np

class RecoveryState(Enum):
    INITIAL = 0
    REACH = 1
    PLACE = 2
    RECOVERY = 3
    DEAD = 4
    SUCCESS = 5
    
TABLE_Z = 0.8
MAX_TIMESTEPS_SAME_STATE = 200

class RecoveryManager:
    def __init__(self, env: PickPlaceWrapper) -> None:
        self.env = env
        self._state = RecoveryState.INITIAL
        self._prev_state = RecoveryState.INITIAL
        self._recovery_goal = np.array([-0.037, -0.104, 0.996])
        self._last_state_timesteps = 0
        self._state_change = 0
        self.state_func_table = {
            RecoveryState.INITIAL: self._next_state_INITIAL,
            RecoveryState.REACH: self._next_state_REACH_,
            RecoveryState.PLACE: self._next_state_PLACE_,
            RecoveryState.RECOVERY: self._next_state_RECOVERY_,
            RecoveryState.DEAD: self._next_state_identity,
            RecoveryState.SUCCESS: self._next_state_identity
        }

    def get_next_state(self, obs) -> RecoveryState:
        tmp_current_state = self.state
        can_pos = self.env.extract_can_pos_from_obs(obs)

        if self._check_success(obs):
            self._state = RecoveryState.SUCCESS
            return self.state

        if self._last_state_timesteps >= MAX_TIMESTEPS_SAME_STATE:
            # we have been too long in the same state
            self._state =  RecoveryState.RECOVERY if self.state != RecoveryState.RECOVERY else  RecoveryState.REACH
            self._last_state_timesteps = 0
            return self._state
        if can_pos[2] < 0.8 or self._state_change >= 50:
            self._state = RecoveryState.DEAD
            return self._state

        if self._prev_state == self._state:
            self._last_state_timesteps += 1
        else:
            self._state_change += 1
            self._last_state_timesteps = 0

        self._prev_state = tmp_current_state

        return self.state_func_table[self._state](obs)
    
    def reset(self):
        self._prev_state = RecoveryState.INITIAL
        self._state = RecoveryState.INITIAL
        self._last_state_timesteps = 0
        self._state_change = 0
        

    def _next_state_INITIAL(self, obs)-> RecoveryState:
        self._state = RecoveryState.REACH
        return self._state
    
    def _next_state_REACH_(self, obs)-> RecoveryState:
        eef_pos = self.env.extract_eef_pos_from_obs(obs)
        can_pos = self.env.extract_can_pos_from_obs(obs)
        eef_to_can = self.env.extract_can_to_eef_dist_from_obs(obs)
        eef_to_can_dist = np.linalg.norm(eef_to_can)
        if eef_to_can_dist < 0.01:
            # reach is done, proceed to place
            self._state = RecoveryState.PLACE
        else:
            if can_pos[2] > eef_pos[2] + 0.1:
                # if the robot is on the same level or under the bin, we need recovery
                self._state = RecoveryState.RECOVERY   
            else:
                self._state = RecoveryState.REACH
        return self._state

    def _next_state_PLACE_(self, obs)-> RecoveryState:
        pick_place_goal = self.env.generate_goal_pick_and_place() # the middle point of the box
        bin_w, bin_h = self.env.get_bin_size()
        can_pos = self.env.extract_can_pos_from_obs(obs)
        eef_to_can = self.env.extract_can_to_eef_dist_from_obs(obs)
        eef_to_can_dist = np.linalg.norm(eef_to_can)

        if (can_pos[0] <= (pick_place_goal[0] + bin_w / 2.0) and
            can_pos[0] >= (pick_place_goal[0] - bin_w / 2.0) and
            can_pos[1] <= (pick_place_goal[1] + bin_h / 2.0) and
            can_pos[1] >= (pick_place_goal[1] - bin_h / 2.0)):
            if can_pos[2] < 0.9:
                self._state = RecoveryState.SUCCESS
            else: 
                self._state = RecoveryState.PLACE
        else: 
            if eef_to_can_dist < 0.1:
                self._state = RecoveryState.PLACE
            else:
                self._state = RecoveryState.RECOVERY
        return self._state

    def _next_state_RECOVERY_(self, obs) -> RecoveryState:
        eef_pos = self.env.extract_eef_pos_from_obs(obs)
        dist_to_goal = np.linalg.norm(eef_pos - self.recovery_goal)
        if dist_to_goal < 0.05:
            self._state = RecoveryState.REACH
        else:
            self._state = RecoveryState.RECOVERY
        return self._state
    
    def _check_success(self, obs):
        pick_place_goal = self.env.generate_goal_pick_and_place() # the middle point of the box
        bin_w, bin_h = self.env.get_bin_size()
        can_pos = self.env.extract_can_pos_from_obs(obs)

        return can_pos[0] <= (pick_place_goal[0] + bin_w / 2.0) and \
            can_pos[0] >= (pick_place_goal[0] - bin_w / 2.0) and \
            can_pos[1] <= (pick_place_goal[1] + bin_h / 2.0) and \
            can_pos[1] >= (pick_place_goal[1] - bin_h / 2.0) and \
            can_pos[2] < 0.9


    def _next_state_identity(self, obs) -> RecoveryState:
        return self._state
    
    @property
    def state(self):
        return self._state
    
    @property
    def recovery_goal(self):
        return self._recovery_goal