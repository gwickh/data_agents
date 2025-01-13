"""
Filename:         agents.py
Author:           Gregory Wickham
Created:          2024-01-14
Version:          0.1
Date modified:    2025-01-13
Description:      LLM-powered agents to perform data cleaning, analysis and visualisation
"""

from crewai import Agent, Task
import pandas as pd
import openai
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

# Set OpenAI API Key

class PlotUtils:
    """
    Functions to read CSV files, create output directory and save timestamped plots.
    """
    
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        """Check file existence and read the CSV."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        return pd.read_csv(file_path)
    
    @staticmethod
    def create_output_directory(output: str):
        """Ensures the output directory exists."""
        if not os.path.exists(output):
            os.makedirs(output)
    
    @staticmethod
    def save_plot(filename: str):
        """Save the plot with a timestamp to avoid overwriting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{timestamp}_{filename}")
    
# Agent 1: Load dataset
class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def run(self):
        data = pd.read_csv(self.dataset_path)
        return data.head(5).to_csv(index=False)


# Agent 2: Generate visualization script
class VisualisationAgent:
    def __init__(self, dataset_path, api_key, model="gpt-4"):
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.model = model
        self.output_dir = "output"
        self.script_name = "visualize.py"

    def create_output_directory(self):
        """Ensures the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_dataset_preview(self, num_rows=5):
        """Loads the dataset and returns a preview."""
        try:
            data = pd.read_csv(self.dataset_path)
            return data.head(num_rows).to_csv(index=False)
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    def generate_visualization_script(self):
        """Uses GPT-4 to generate a visualization script."""
        # Load a preview of the dataset to pass to GPT
        dataset_preview = self.load_dataset_preview()
        dataset_path = os.path.basename(self.dataset_path)
        
        # GPT-4 prompt
        prompt = f"""
            You are a Python data analysis assistant. Generate a Python script to visualize the following dataset.
            The dataset preview is:
            {dataset_preview}

            The script must:
            1. Load {dataset_path} with pandas.
            2. Use seaborn/matplotlib to create a plot visualising the data.
            3. Save plots to an `output` directory.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"]  # Return the script

# Execute script
class ScriptExecutorAgent:
    def __init__(self, script_name):
        self.script_name = script_name

    def run(self):
        import subprocess
        subprocess.run(["python", self.script_name])
        
class CrewAIVisualizationWorkflow:
    def __init__(self, api_key_path, dataset_path):
        self.api_key_path = api_key_path
        self.dataset_path = dataset_path

    def create_workflow(self):
        # Initialize agents
        dataset_loader = DatasetLoader(self.dataset_path)
        gpt_generator = VisualisationAgent(self.api_key_path)
        script_executor = ScriptExecutorAgent()

        # Set up the CrewAI workflow
        crew = CrewAI()
        crew.add_task("Load Dataset", dataset_loader.run)
        crew.add_task(
            "Generate Script",
            gpt_generator.run,
            requires=["Load Dataset"],
            inputs={"dataset_preview": "Load Dataset", "dataset_path": self.dataset_path},
        )
        crew.add_task(
            "Execute Script",
            script_executor.run,
            requires=["Generate Script"],
            inputs={"script_content": "Generate Script"},
        )
        return crew

    def run(self):
        workflow = self.create_workflow()
        workflow.execute()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CrewAI-based visualization workflow.")
    parser.add_argument("--api_key", required=True, help="Path to the text file containing the OpenAI API key.")
    parser.add_argument("--data", required=True, help="Path to the dataset in CSV format.")
    args = parser.parse_args()

    # Run the workflow
    workflow = CrewAIVisualizationWorkflow(api_key_path=args.api_key, dataset_path=args.data)
    workflow.run()