from utils.env_wrapper import PendulumWrapper, LunarLanderContinuousWrapper, BipedalWalkerWrapper
from utils.tank_env import Tank_Env
import tensorflow as tf

class train_params:
    MODE = 'OSMC-DDPG'
    # MODE = 'OSMC-D4PG'
    print('MODE: {}'.format(MODE))

    # Environment parameters
    ENV = 'Tank_Env_2D-2crit' # 'Tank_Env'  # 'Pendulum-v0'                     # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RENDER = False                          # Whether or not to display the environment on the screen during training
    RANDOM_SEED = 99999999                  # Random seed for reproducability
    NUM_AGENTS = 4                         # Number of distributed agents to run simultaneously

    # Create dummy environment to get all environment params
    if ENV == 'Pendulum-v0':
        dummy_env = PendulumWrapper()
    elif ENV == 'LunarLanderContinuous-v2':
        dummy_env = LunarLanderContinuousWrapper()
    elif ENV == 'BipedalWalker-v2':
        dummy_env = BipedalWalkerWrapper()
    elif ENV == 'BipedalWalkerHardcore-v2':
        dummy_env = BipedalWalkerWrapper(hardcore=True)
    elif ENV == 'Tank_Env_2D-2crit':
        dummy_env = Tank_Env('2D-2crit', None)
    else:
        raise Exception('Chosen environment does not have an environment wrapper defined. Please choose an environment with an environment wrapper defined, or create a wrapper for this environment in utils.env_wrapper.py')

    STATE_DIMS = dummy_env.get_state_dims()
    print('PARAMS STATE_DIMS {}'.format(STATE_DIMS))
    STATE_BOUND_LOW, STATE_BOUND_HIGH = dummy_env.get_state_bounds()
    print('STATE_BOUND_LOW: {}, STATE_BOUND_HIGH: {}'.format(STATE_BOUND_LOW, STATE_BOUND_HIGH))
    ACTION_DIMS = dummy_env.get_action_dims()
    print('PARAMS ACTION_DIMS {}'.format(ACTION_DIMS))
    ACTION_BOUND_LOW, ACTION_BOUND_HIGH = dummy_env.get_action_bounds()
    print('ACTION_BOUND_LOW: {}, ACTION_BOUND_HIGH: {}'.format(ACTION_BOUND_LOW, ACTION_BOUND_HIGH))
    V_MIN = dummy_env.v_min
    print('PARAMS V_MIN: {} {}'.format(type(V_MIN), V_MIN))
    V_MAX = dummy_env.v_max
    print('PARAMS V_MAX: {} {}'.format(type(V_MAX), V_MAX))
    del dummy_env

    # Training parameters
    BATCH_SIZE = 64 #256 default, 128 OSTC
    NUM_STEPS_TRAIN = 1000000        # Number of steps to train for 1000000
    MAX_EP_LENGTH = 10000           # Maximum number of steps per episode
    REPLAY_MEM_SIZE = 1000000       # Soft maximum capacity of replay memory
    REPLAY_MEM_REMOVE_STEP = 200    # Check replay memory every REPLAY_MEM_REMOVE_STEP training steps and remove samples over REPLAY_MEM_SIZE capacity
    PRIORITY_ALPHA = 0.6            # Controls the randomness vs prioritisation of the prioritised sampling (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    PRIORITY_BETA_START = 0.4 # 0.4       # Starting value of beta - controls to what degree IS weights influence the gradient updates to correct for the bias introduced by priority sampling (0 - no correction, 1 - full correction)
    PRIORITY_BETA_END = 1.0         # Beta will be linearly annealed from its start value to this value throughout training
    PRIORITY_EPSILON = 0.00001      # Small value to be added to updated priorities to ensure no sample has a probability of 0 of being chosen
    NOISE_STD = 0.1 #0.3                 # STD to apply to Gaussian noise
    NOISE_DECAY = 0.9999            # Decay noise throughout training by scaling by noise_decay**training_step
    DISCOUNT_RATE = 0 #0.99 D4PG           # Discount rate (gamma) for future rewards
    N_STEP_RETURNS = 5              # Number of future steps to collect experiences for N-step returns
    UPDATE_AGENT_EP = 100 #10            # Agent gets latest parameters from learner every update_agent_ep episodes 10
    UPDATE_TARGET_EP = 10

    # Network parameters
    CRITIC_LR = 0.0003 #3e-4  # 0.0001 default, 1e-3 DDPG, 3e-4 OSTC
    ACTOR_LR = 0.0001
    CRITIC_L2_LAMBDA = 0.0          # Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed 0 default, 1e-2 DDPG
    DENSE1_SIZE = 512               # Size of first hidden layer in networks
    DENSE2_SIZE = 512               # Size of second hidden layer in networks
    ADDED_SIZE = 512
    FINAL_LAYER_INIT = 0.003        # Initialise networks' final layer weights in range +/-final_layer_init
    NUM_ATOMS = 128                  # Number of atoms in output layer of distributional critic
    TAU = 0.01 # 0.001                     # Parameter for soft target network updates
    USE_BATCH_NORM = False          # Whether or not to use batch normalisation in the networks
    M1_REW_SCALE = 1.0
    M2_REW_SCALE = 1.0

    # Files/Directories
    SAVE_CKPT_STEP = 10000                    # Save checkpoint every save_ckpt_step training steps 10000
    CKPT_DIR = './ckpts/' + ENV             # Directory for saving/loading checkpoints
    # CKPT_FILE = None                        # Checkpoint file to load and resume training from (if None, train from scratch)
    CKPT_NUM = None #'660000-66'
    LOG_DIR = './logs/train/' + ENV         # Directory for saving Tensorboard logs (if None, do not save logs)


class test_params:

    # Environment parameters
    ENV = train_params.ENV                                  # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RENDER = False                                          # Whether or not to display the environment on the screen during testing
    RANDOM_SEED = 999999                                    # Random seed for reproducability

    # Testing parameters
    NUM_EPS_TEST = 100                                      # Number of episodes to test for
    MAX_EP_LENGTH = 10000                                   # Maximum number of steps per episode

    # Files/directories
    CKPT_DIR = './ckpts/' + ENV                             # Directory for saving/loading checkpoints
    CKPT_FILE = 'Pendulum-v0_actor_90000-3.510e-06'                                        # Checkpoint file to load and test (if None, load latest ckpt)
    RESULTS_DIR = './test_results'                          # Directory for saving txt file of results (if None, do not save results)
    LOG_DIR = './logs/test/' + ENV                          # Directory for saving Tensorboard logs (if None, do not save logs)


class play_params:

    # Environment parameters
    ENV = train_params.ENV                                  # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RANDOM_SEED = 999999                                    # Random seed for reproducability

    # Play parameters
    NUM_EPS_PLAY = 5                                        # Number of episodes to play for
    MAX_EP_LENGTH = 10000                                   # Maximum number of steps per episode

    # Files/directories
    CKPT_DIR = './ckpts/' + ENV                             # Directory for saving/loading checkpoints
    CKPT_FILE = 'Pendulum-v0.ckpt-660000'                                        # Checkpoint file to load and run (if None, load latest ckpt)
    RECORD_DIR = './video'                                  # Directory to store recorded gif of gameplay (if None, do not record)



