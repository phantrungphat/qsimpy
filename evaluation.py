from ray.rllib.algorithms import Algorithm
from env_creator import qsimpy_env_creator
from ray.tune.registry import register_env
import os, csv


class Evaluation:
    def __init__(self, algo_path, num_iterations=100):

        self.algo_path = algo_path
        self.num_iterations = num_iterations

        register_env("QSimPyEnv", qsimpy_env_creator)
        self.model = Algorithm.from_checkpoint(self.algo_path)
        env_config={
            "obs_filter": "rescale_-1_1",
            "reward_filter": None,
            "dataset": "qdataset/qsimpyds_1000_sub_36.csv",
        }
        self.env = qsimpy_env_creator(env_config)
        self.results = []

    def run(self):
        """
        Run the PPO algorithm solution.
        """

        self.results = []
        # Reset the subset of QTasks 
        self.env.round = 1
        
        for _ in range(self.num_iterations):

            # Initialize the temporary array to store the results of the QTasks execution for each episode
            arr_temp = {
                "total_completion_time": 0.0,
                "rescheduling_count": 0.0
            }
            terminated = False

            # Reset the environment and setup the quantum resources
            obs, _ = self.env.reset()
            self.env.setup_quantum_resources()

            while not terminated:
                # Get the action with the given control
                action = self.model.compute_single_action(obs, state=None, explore=False)
                obs, reward, terminated, done, info = self.env.step(action)
                
                if reward > 0:
                    """Get the results of the QTask execution

                    Values:
                        - Total Completion Time: waiting_time + execution_time
                        - Rescheduling Count: rescheduling_count
                    """

                    arr_temp["total_completion_time"] += info["scheduled_qtask"].waiting_time + info["scheduled_qtask"].execution_time
                    arr_temp["rescheduling_count"] += info["scheduled_qtask"].rescheduling_count

            # self.env.qsp_env.run()

            # Final results of the episode
            self.results.append(arr_temp)
            
        # Save the results to a CSV file
        self._save_to_csv("PPO")

    def _save_to_csv(self, control) -> None:
        """
        Save values and episodes to a CSV file.
        """

        file_name = "./evaluation/" + control + "/"

        if not os.path.exists(file_name):
            os.makedirs(file_name)

        file_name += "result.csv"
        # Open the CSV file in write mode
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Interation', 'Finish Time', 'Reward'])
            
            # Write the data
            for i in range(len(self.results)):
                writer.writerow([i, self.results[i]['finish_time'], self.results[i]['reward']])
        print("CSV file saved to " + file_name)

if __name__ == "__main__":
    path = "results/PPO_qce_1000/PPO_QSimPyEnv_1f6bb_00000_0_2024-08-12_16-56-19/checkpoint_000009"
    eval = Evaluation(algo_path=path, num_iterations=10)
    eval.run()
