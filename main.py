from environment import GymEnvironment
from ddpg_agent import DDPGAgent
import numpy as np
import sys
import random
import os
import json
import pdb

def main():
    # Get the path to the configs
    model_path = sys.argv[1]

    # Parse the config file
    conf_path = os.path.join(model_path, 'config.json')
    json_data = open(conf_path).read()
    conf = json.loads(json_data)

    # Update the relative path
    conf["agent"]["actor_path"]         = os.path.join(model_path, conf["agent"]["actor_path"] )
    conf["agent"]["critic_path"]        = os.path.join(model_path, conf["agent"]["critic_path"] )
    conf["agent"]["hallucinator_path"]  = os.path.join(model_path, conf["agent"]["critic_path"] )

    # Run the training algorithm
    run = Runner(conf['env'], conf['agent'])
    run.train(conf['train'])
    run.test(conf['test'])

class Runner:
    def __init__(self, env_config, agent_config):
        self.env = GymEnvironment(name = env_config["name"])
        self.agent = DDPGAgent(action_size = self.env.action_size[0],
                                state_size = self.env.obs_size[0],
                                **agent_config)

    def train(self, train_config, fill_replay = True):
        '''
        Traijs the policy, the critic and the hallucinator.
        '''
        # Run on per episode basis
        ma_reward = 0

        train_episodes = train_config['episodes']

        # Train for some number of episodes
        for step in range(train_episodes):
            # Start a new episode
            self.env.new_episode()
            episode_reward = 0
            episode_done = False

            # Keep going until the episode is over
            while(!episode_done):
                # Get observation from environment
                cur_obs = self.env.cur_obs
                # Query agent for action to be performed
                cur_action = np.squeeze(self.agent.get_next_action(cur_obs), axis=0)

                '''
                if (any(np.isnan(cur_obs))):
                    pdb.set_trace()
                '''

                # Step the simulation
                next_state, reward, done = self.env.next_obs(cur_action, render = True)

                # Aggregate the reward
                episode_reward += reward

                # Update wether we are done or not
                episode_done = episode_done or done

            # Once we are done with the episode, store (policy, reward)
            self.policy_buffer.store_tuple(self.agent.get_weights(), episode_reward)

            # Now train critic and hallucinator

            # Now optimize the policy using the critic, hallucinator

            '''
            ma_reward = ma_reward*0.99 + reward*0.01
            if(step % 1000):
                print(cur_obs, ' ', cur_action, 'Reward:', ma_reward)
                print('Eps',self.agent.epsilon)
            '''

    def test(self, test_config):
        '''
        Simply runs the trained policy in the environment.
        '''
        test_steps = test_config['steps']

        temp_reward = 0
        temp_done = False
        for step in range(start_train):
            cur_obs = self.env.cur_obs
            cur_obs = np.concatenate((cur_obs,np.array([temp_reward])))
            cur_action = self.agent.get_next_action(cur_obs)
            next_state, reward, done = self.env.next_obs(cur_action, render = True)

if __name__ == "__main__":
    main()
