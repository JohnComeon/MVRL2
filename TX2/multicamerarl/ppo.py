import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


def transform_buffer(buff):
    left_bottom_batch, central_bottom_batch, right_bottom_batch,speed_batch, command_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch = [], [], [], [], [], [], [], [],[],[]
    left_bottom, central_bottom,right_bottom, speed_temp , command_temp = [], [], [],[],[]

    for e in buff:
        for state in e[0]:
            left_bottom.append(state[0])
            central_bottom.append(state[1])
            right_bottom.append(state[2])
            speed_temp.append(state[3])
            command_temp.append(state[4])
        left_bottom_batch.append(left_bottom)
        central_bottom_batch.append(central_bottom)
        right_bottom_batch.append(right_bottom)
        speed_batch.append(speed_temp)
        command_batch.append(command_temp)
        left_bottom = []
        central_bottom = []
        right_bottom = []
        speed_temp = []
        command_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    left_bottom_batch = np.asarray(left_bottom_batch)
    central_bottom_batch = np.asarray(central_bottom_batch)
    right_bottom_batch = np.asarray(right_bottom_batch)
    speed_batch = np.asarray(speed_batch)
    command_batch = np.asarray(command_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return left_bottom_batch, central_bottom_batch, right_bottom_batch, speed_batch, command_batch,a_batch, r_batch, d_batch, l_batch, v_batch


def generate_action(env, state_list, vl_policy, action_bound):
    if env.index == 0:
        left_list, central_list, right_list, speed_list ,command_list= [], [], [], [], []
        for i in state_list:
            left_list.append(i[0])
            central_list.append(i[1])
            right_list.append(i[2])
            speed_list.append(i[3])
            command_list.append(i[4])

        left_list = np.asarray(left_list)
        central_list = np.asarray(central_list)
        right_list = np.asarray(right_list)
        speed_list = np.asarray(speed_list)
        command_list = np.asarray(command_list)

        left_list = Variable(torch.from_numpy(left_list)).float().cuda()
        central_list = Variable(torch.from_numpy(central_list)).float().cuda()
        right_list = Variable(torch.from_numpy(right_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
        command_list = Variable(torch.from_numpy(command_list)).float().cuda()

        v, a, logprob, mean, weights = vl_policy(left_list, central_list,right_list,command_list)
        
        v, a, logprob, weights = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy(),weights.data.cpu().numpy() #, mean.data.cpu().numpy(),std.data.cpu().numpy()

        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
   
       
        
    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action,weights

def generate_action_no_sampling(env, state_list, vl_policy, action_bound):
    if env.index == 0:
        left_list, central_list, right_list, speed_list, command_list= [], [], [],[], []
        for i in state_list:
            left_list.append(i[0])
            central_list.append(i[1])
            right_list.append(i[2])
            speed_list.append(i[3])
            command_list.append(i[4])

        left_list = np.asarray(left_list)
        central_list = np.asarray(central_list)
        right_list = np.asarray(right_list)
        speed_list = np.asarray(speed_list)
        command_list = np.asarray(command_list)

        left_list = Variable(torch.from_numpy(left_list)).float().cuda()
        central_list = Variable(torch.from_numpy(central_list)).float().cuda()
        right_list = Variable(torch.from_numpy(right_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
        command_list = Variable(torch.from_numpy(command_list)).float().cuda()

        v, a, logprob, mean,weights = vl_policy(left_list, central_list,right_list,command_list)
        
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
        
        mean = mean.data.cpu().numpy()
        scaled_action1 = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
    else:
        mean = None
        scaled_action = None

    return v, a, logprob, scaled_action,scaled_action1



def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    returns = np.zeros((num_step + 1, num_env))
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(num_step)):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        targets[t, :] = gae + values[t, :]

    advs = targets - values[:-1, :]
    return targets, advs



def ppo_update_stage1 (vl_policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=12, size=160,act_size=4):
    left, central , right, speeds, commands, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    left = left.reshape((num_step*num_env, 1,size))
    #print(rgb.shape)
    central = central.reshape((num_step*num_env, 1,size))
    right =right.reshape((num_step*num_env,1, size))
    #print(rgb.shape)
    speeds = speeds.reshape((num_step*num_env, 2))
    commands = commands.reshape((num_step * num_env, 4))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_left = Variable(torch.from_numpy(left[index])).float().cuda()
            sampled_central = Variable(torch.from_numpy(central[index])).float().cuda()
            sampled_right = Variable(torch.from_numpy(right[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()
            sampled_commands = Variable(torch.from_numpy(commands[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            
            new_value, new_logprob, dist_entropy = vl_policy.evaluate_actions(sampled_left,sampled_central,sampled_right , sampled_commands, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = 20 *F.mse_loss(new_value, sampled_targets)

            loss = policy_loss +  value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))
    	    #writer.add_scalar('P_LOSS',info_p_loss)
	    #writer.add_scalar('V_LOSS',info_v_loss)
       	    #writer.add_scalar('entropy',info_entropy)

    print('update')

