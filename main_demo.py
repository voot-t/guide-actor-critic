
import argparse
import os
import platform
import time
import numpy as np
import tensorflow as tf 
import random as rn
from keras import backend as K 
import gym
import mujoco_py
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #Do not show gpu information when running tensorflow

from GAC_learner import GAC_learner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo policy.")
    parser.add_argument("-s", "--seed", dest="seed", default=0, type=int, help="Random seed")
    
    parser.add_argument("-e", "--env", dest="env_name", default="Pendulum-v0", type=str, help="Name of a gym environment. Default is Pendulum.")
    parser.add_argument("-tstep", "--time_step", dest="time_step", default=1000, type=int, help="Maximum time steps in one episode")
    
    parser.add_argument("-lq", "--lr_q", dest="lr_q",         default=0.001, type=float, help="Learning rate the Q-network")
    parser.add_argument("-lp", "--lr_p", dest="lr_policy",    default=0.0001, type=float, help="Learning rate for the policy network")
    parser.add_argument("-rq", "--reg_q", dest="reg_q",      default=0, type=float, help="Regularizer for the Q-network")
    parser.add_argument("-rp", "--reg_p", dest="reg_policy", default=0, type=float, help="Regularizer for the Q-network")
    
    parser.add_argument("-eps", "--eps", dest="epsilon",        default=0.0001, type=float, help="KL upper-bound")
    parser.add_argument("-nt", "--n_taylor", dest="n_taylor", default=1, type=int, help="Number of samples for Taylor approximation. 0=GAC-0, 1=GAC-1")

    parser.add_argument("-itest", "--itest", dest="iter_test", default=1000, type=float, help="Test iteration")

    args = parser.parse_args()
    seed = args.seed
    env_dict = {-2 : "BipedalWalker-v2",
                -1 : "LunarLanderContinuous-v2", 
                0 : "Pendulum-v0",
                1 : "InvertedPendulum-v1",
                2 : "HalfCheetah-v1",
                3 : "Reacher-v1",
                4 : "Swimmer-v1",
                5 : "Ant-v1",
                6 : "Hopper-v1",
                7 : "Walker2d-v1",
                8 : "InvertedDoublePendulum-v1",
                9 : "Humanoid-v1",
                10: "HumanoidStandup-v1",
    }
    #env_idx     = args.env_idx
    #env_name    = env_dict[env_idx]
    env_name    = args.env_name
    T_max       = args.time_step

    epsilon     = args.epsilon
    lr_q        = args.lr_q
    lr_policy   = args.lr_policy
    reg_q       = args.reg_q 
    reg_policy  = args.reg_policy
    n_taylor    = args.n_taylor

    iter_test       = args.iter_test

    render = 1
    N_test = 10
    save_video = False


    np.random.seed(seed)
    tf.set_random_seed(seed)
    sess = tf.Session()
    K.set_session(sess)
        
    env_test = gym.make(env_name)
    env_test.seed(seed)

    if save_video:
        try:
            import pathlib
            pathlib.Path("./Video/" + env_name).mkdir(parents=True, exist_ok=True) 
            video_relative_path = "./Video/" + env_name + "/"

            # Change from N_test-1 to 0 to save video of every episodes
            env_test = gym.wrappers.Monitor(env_test, video_relative_path, \
                video_callable=lambda episode_id: episode_id%1==(N_test-1), force =True)
        except:
            print("Cannot create video directories. Video will not be saved.")
            save_video = False

    s_space = env_test.observation_space
    a_space = env_test.action_space
    ds = np.shape(s_space)[0]
    da = np.shape(a_space)[0]
    s_space_high = s_space.high
    s_space_low = s_space.low
    a_space_high = a_space.high
    a_space_low = a_space.low

    # Set the directory path to a keras model 
    expname = "GAC-%d_%s_eps%0.4f_lrq%0.4f_lrp%0.4f_rq%0.4f_rp%0.4f_s%d" \
                % (n_taylor, env_name, epsilon, lr_q, lr_policy, reg_q, reg_policy, seed)
    dir_path = os.path.dirname(os.path.realpath(__file__)) 
    model_path = dir_path + "\\Model\\" + env_name
    filename = os.path.join(model_path, expname + ("_mean_I%d.h5" % iter_test))

    # Construct agent for testing. We only need to specified action bounds.
    learner = GAC_learner(ds=ds, da=da, sess=sess, action_bnds =[a_space_low[0], a_space_high[0]])

    # Load the model of policy (covariance not included)           
    learner.deep_mean_model.load_weights(filename) 

    ret_te = np.zeros((1, N_test))
    np.random.seed(seed)               
    t_te_sum = 0
    for nn in range(0, N_test):
        state_te = env_test.reset() 
        for t_te in range(0, T_max):
            if render:
                env_test.render()
                time.sleep(0.01)    #to slowdown rendering

            action_te = learner.get_action(state_te)
            next_state_te, reward_te, done_te, info_te = env_test.step(np.clip(action_te, a_min=a_space_low, a_max=a_space_high))  

            state_te = next_state_te
            ret_te[0, nn]  = ret_te[0, nn]  + reward_te 
            t_te_sum = t_te_sum + 1
            if done_te:
                break 
            
        result = "Episode %d, return %0.4f, length: %d" % (nn, ret_te[0, nn], t_te+1) 
        print(result)

    ret_mean = np.mean(ret_te[0, :])
    ret_std = np.std(ret_te[0, :]) / np.sqrt(N_test)
    t_run = t_te_sum / N_test

    print("Averaged return: %f(%f), averaged timesteps %f" % (ret_mean, ret_std, t_run))

    K.clear_session()

