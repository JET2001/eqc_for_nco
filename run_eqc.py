import tensorflow as tf

from src.utils.limit_thread_usage import set_thread_usage_limit
set_thread_usage_limit(10, tf)

from src.model.q_learning import CircuitType
from run import run_tsp
from config import BASE_PATH


hyperparams = {
    'n_vars': 5, # length of tour
    'episodes': 5000, # total episodes
    'batch_size': 10, # DQN batch size
    'epsilon': 1, # epsilon greedy policy? 
    'epsilon_decay': 0.99, # eps greedy decay
    'epsilon_min': 0.01, # eps greedy min
    'gamma': 0.9, # discounted factor of return
    'update_after': 10,  
    'update_target_after': 30, 
    'learning_rate_in': 0.00001, # 1e-5
    'n_layers': 1,
    'epsilon_schedule': 'fast',
    'memory_length': 10000,
    'num_instances': 100,
    'circuit_type': CircuitType.EQC,
    'data_path': BASE_PATH + 'tsp/tsp_5_train/tsp_5_reduced_train.pickle',
    'repetitions': 1,
    'save': False,
    'test': True
}


if __name__ == '__main__':
    run_tsp(hyperparams, '/save_path/')
