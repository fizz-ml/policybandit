import torch as t
from torch.autograd import Variable as V
from torch import FloatTensor as FT
import numpy as np

class SimpleTrainer:
    def __init__(env,critic,hallucinator,policy_buffer,policy):
        self.env = env
        self.hallucinator = hallucinator
        self.critic = critic
        self.policy_buffer = policy_buffer
        self.policy_c = policy_c

    def train(train_steps,sample_steps,opt_steps):
        in_dim=self.env.state_size()
        out_dim=self.env.action_size()

        cur_policy = self.policy_c(in_dim,out_dim)
        for i in range(train_steps):
            reward = self.sample_episode(cur_policy)
            self.policy_buffer.put(policy.state_dict(),reward)
            self.train_critic_hallucinator(sample_steps)
            self.train_policy(opt_steps)

    def sample_episode(self,policy,n=1,skip = 100):
        policy = self.policy_c(state_dict)
        done = False
        total_reward = 0 
        for i in range(n):
            cur_obs = self.env.new_episode()
            t = 0
            while not done:
                display = (t % 100 == 0)
                cur_action = policy.forward(cur_obs)
                cur_obs,cur_reward,done = self.env.next_obs(cur_action)
                total_reward += cur_reward
                t += 1

        avg_episode_reward = total_reward / n
        return avg_episode_reward


    def train_critic_hallucinator(self,sample_steps):
        def closure_gen():
            yield (lambda: self.critic.get_prior_llh())

            for state_dict,reward in self.policy_buffer:
                policy = self.policy_c(state_dict)

                def closure():
                    noise=(FT(np.randn(self.noise_dim)))
                    states = self.hallucinator.forward(noise)
                    actions = policy.forward(states)
                    mean = critic(states,actions)[0]
                    lsd = critic(states,actions)[0]
                    llh =  gaussian_llh(meah,lsd,reward)
                    return reward

                yield closure

        params = self.critic.parameter_list() \ 
            + self.halucinator.parameter_list()
           
        sampler = HMCSampler(parameters,closure_gen)
        for i in range(sample_steps):
            sampler.step()

    def train_policy(self,opt_steps):
        state_dict = self.policy_buffer.peak()
        policy = self.policy_c(state_dict)
        
        opt = t.optim.SGD(policy.parmeters())

        def closure():
            noise=(FT(np.randn(self.noise_dim)))
            states = self.hallucinator.forward(noise)
            actions = policy.forward(states)
            reward = critic(states,actions)[0]
            return reward

        for i in range(opt_steps):
            opt.zero_grad()
            opt.step(closure)

        return policy.state_dict()

def gaussian_llh(mean,log_std_dev,reward):
    llh = -(mean-reward)**2 - 2*log_std_dev
    return llh
