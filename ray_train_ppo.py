import argparse

import ray
from ray import tune, air, train
from ray.tune.registry import register_env
from env_creator import qsimpy_env_creator
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.algorithms.ppo import PPOConfig
import os


tf1, tf, tfv = try_import_tf()
parser = argparse.ArgumentParser()

parser.add_argument("--num-cpus", type=int, default=0)

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)

parser.add_argument(
    "--stop-iters", type=int, default=100, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)

if __name__ == "__main__":
    args = parser.parse_args()

    # ray.init(num_cpus=args.num_cpus or None)

    register_env("QSimPyEnv", qsimpy_env_creator)

    config = (
        PPOConfig()
        .framework(framework=args.framework)
        .environment(
            env="QSimPyEnv",
            env_config={
                "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
            },
        )
        .env_runners(num_env_runners=4)
        .training( # you can use tune.grid_search to search for the best parameters
            gamma=0.99,
            lambda_=0.95,
            lr=0.1,
            clip_param=0.2,
            kl_coeff=0.3,
            sgd_minibatch_size=32, 
            num_sgd_iter=10,
        )
        
    )

    stop_config = {
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }
    
    # Get the absolute path of the current directory
    current_directory = os.getcwd()

    # Append the "result" folder to the current directory path
    result_directory = os.path.join(current_directory, "results")

    # Create the storage_path with the "file://" scheme
    storage_path = f"file://{result_directory}"

    results = tune.Tuner(
        "PPO", 
        run_config=air.RunConfig(
            stop=stop_config,
            # Save checkpoints every 10 iterations.
            checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
            storage_path=storage_path, 
            name="PPO_qce_1000"
        ),
        param_space=config.to_dict(),
    ).fit()

    ray.shutdown()