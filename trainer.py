import torch as t
from torch.autograd import Variable as V
from torch import FloatTensor as FT
import numpy as np
from bayestorch.hmc import HMCSampler

class SimpleTrainer:
    def __init__(self, env,critic,hallucinator,policy_buffer,policy_c, noise_dim):
        self.env = env
        self.hallucinator = hallucinator
        self.critic = critic
        self.policy_buffer = policy_buffer
        self.policy_c = policy_c
        self.noise_dim = noise_dim

    def train(self, train_steps,sample_steps,opt_steps):
        in_dim=self.env.obs_size
        out_dim=self.env.action_size

        cur_policy = self.policy_c(in_dim,out_dim)
        for i in range(train_steps):
            reward = self.sample_episode(cur_policy)
            self.policy_buffer.put(cur_policy.state_dict(),reward)
            self.train_critic_hallucinator(sample_steps)
            self.train_policy(opt_steps)
            print("#################################################")

    def sample_episode(self, policy,n=10,skip = 3):
        total_reward = 0
        for i in range(n):
            cur_obs = self.env.new_episode()
            t = 0
            done = False
            while not done:
                cur_obs = V(FT(cur_obs)).unsqueeze(0)
                display = (t % skip == 0)
                cur_action = policy.forward(cur_obs).data.cpu().numpy()
                cur_obs,cur_reward,done = self.env.next_obs(cur_action.squeeze(0), render = False)
                total_reward += cur_reward
                t += 1

        avg_episode_reward = total_reward / n
        print("AVG_REWARD:",avg_episode_reward)
        return avg_episode_reward


    def train_critic_hallucinator(self,sample_steps):
        def closure_gen():
            prior_c = self.critic.get_prior_llh()
            prior_h = self.hallucinator.get_prior_llh()
            prior = prior_c + prior_h
            yield (lambda: -prior)

            for state_dict,reward in self.policy_buffer:
                policy = self.policy_c(self.env.obs_size, self.env.action_size)
                policy.load_state_dict(state_dict)

                def closure():
                    noise=V(FT(np.random.randn(self.noise_dim)))
                    states = self.hallucinator.forward(noise.unsqueeze(0))

                    # Concatenating dimensions of bath(which is currently 1) and dimensions of
                    states = states.view(states.size(0)*self.hallucinator.n, -1)
                    actions = policy.forward(states)
                    actions = actions.view(1,-1)
                    states = states.view(1,-1)

                    mean = self.critic(states,actions)[0]
                    lsd = self.critic(states,actions)[0]
                    llh =  gaussian_llh(mean,lsd,reward)
                    return -llh

                yield closure

        params = self.critic.parameter_list() \
            + self.hallucinator.parameter_list()

        sampler = HMCSampler(params)
        for i in range(sample_steps):
            sampler.step(closure_gen)

    def train_policy(self,opt_steps):
        state_dict, _ = self.policy_buffer.peek()
        policy = self.policy_c(self.env.obs_size, self.env.action_size)
        policy.load_state_dict(state_dict)

        opt = t.optim.Adam(policy.parameters(), lr=0.1)

        # This is bad just have one goddamnit
        def closure():
            noise=V(FT(np.random.randn(self.noise_dim)))
            states = self.hallucinator.forward(noise.unsqueeze(0))

            # Concatenating dimensions of bath(which is currently 1) and dimensions of
            states = states.view(states.size(0)*self.hallucinator.n, -1)
            #print(states)
            actions = policy.forward(states)
            actions = actions.view(1,-1)
            states = states.view(1,-1)
            reward = self.critic(states,actions)[0]
            return -reward

        for i in range(opt_steps):
            opt.zero_grad()
            x = closure()
            print("----------------------------------")
            x.backward()
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print([x.grad for x in policy.parameters()])
            print("**********************************")
            opt.step()
            if i % 500 == 0:
                reward = closure()
                print("EXP_REWARD",-reward.data.cpu().numpy())

        return policy.state_dict()

def gaussian_llh(mean,log_std_dev,reward):
    llh = -(mean-reward)**2 - 2*log_std_dev
    return llh
