from environment import GymEnvironment
import numpy as np
import sys
import random
import os
import json
from critic import SimpleCritic
from policy import Policy
from hallucinator import SimpleHallucinator
from trainer import SimpleTrainer
from policy_buffer import PolicyBuffer

def main():
    # Get the path to the configs
    model_path = sys.argv[1]

    # Parse the config file
    conf_path = os.path.join(model_path, 'config.json')
    json_data = open(conf_path).read()
    conf = json.loads(json_data)

    '''
    # Update the relative path
    conf["agent"]["actor_path"]         = os.path.join(model_path, conf["agent"]["actor_path"] )
    conf["agent"]["critic_path"]        = os.path.join(model_path, conf["agent"]["critic_path"] )
    conf["agent"]["hallucinator_path"]  = os.path.join(model_path, conf["agent"]["critic_path"] )
    '''

    # Run the training algorithm
    run = Runner(conf['env'], conf['agent'])
    run.train(conf['train'])
    #run.test(conf['test'])

class Runner:
    def __init__(self, env_config, agent_config):
        self.n = 10
        self.noise_dim = 2
        self.env = GymEnvironment(name = env_config["name"])
        self.critic = SimpleCritic(self.n, self.env.obs_size, self.env.action_size)
        self.hallucinator = SimpleHallucinator(self.n, self.env.obs_size, self.noise_dim)
        self.policy_buffer = PolicyBuffer()
        self.policy_c = Policy
        self.trainer = SimpleTrainer(self.env, self.critic, self.hallucinator, self.policy_buffer, self.policy_c, self.noise_dim)

    def train(self, train_config, fill_replay = True):
        train_steps = train_config['steps']
        self.trainer.train(train_steps, 2, 2)

if __name__ == "__main__":
    main()
