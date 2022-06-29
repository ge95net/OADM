import argparse
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.CoppeliaAgent.CoppeliaAgent import CoppeliaAgent
import sys
import configparser

class Human(Agent):
    counter = 0
    def __init__(self, config, section,client):
        super().__init__(config, section)
        self.id = self.counter
        self.human_num = 5
        self.set_client(client,id=self.id)
        self.__class__.counter+=1
        if self.__class__.counter >= self.human_num:
            self.__class__.counter = 0

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def set_client(self,client,name=None,id=None):
        self.agent = CoppeliaAgent(client,name,id)