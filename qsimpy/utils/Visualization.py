import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

class Visualization:
    def sumerize_results(num_episodes, values, label) -> None:
        """
        Summarize the results of the episodes.
        """
        print("Results Summary for " + label + "solution:")
        print(f"Number of Episodes: {num_episodes}")
        print(f"Total Waiting Time: {sum(values['Total Completion Time'])}")
        print(f"Average Rescheduling Count: {sum(values['Rescheduling Count']) / num_episodes}")
        
    def plot_results(paths, num_episodes=100) -> None:
        """
        Plot the results of the episodes.
        """
        for path in paths:
            df1 = pd.read_csv(path['path'])

            plt.plot(df1['Episode'], df1['Total Completion Time'], ".-", color=path['color'], label=path['label'])

            Visualization.sumerize_results(num_episodes, df1, path['label'])
        
        plt.ylabel('Total Completion Time')
        plt.xlabel('Evaluation Episode')
        plt.legend(loc=2)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
        plt.show()

    