import openai
import re
import json
from openai import OpenAI
import os
from datetime import datetime

client = OpenAI()

def call_gpt(cur_prompt, stop=None, model="gpt-4o-mini"):
    reasoner_messages = [
        {
            "role": "user",
            "content": cur_prompt
        },
    ]
    completion = openai.chat.completions.create(
        model=model,
        messages=reasoner_messages,
    )
    returned = completion.choices[0].message.content
    return returned


def extract_ans(input_string):
    """
    Extract all contents inside angle brackets <> from the given string.

    Args:
        input_string (str): The input string containing angle brackets.

    Returns:
        list: A list of strings extracted from inside the angle brackets.
    """
    # Use regex to find all matches between angle brackets and trim extra whitespace
    matches = re.findall(r'<\s*(.*?)\s*>', input_string)
    return matches[0]


class DataHandler:
    def __init__(self, task_description, result_folder):
        # Get the current time and format it as the folder name
        self.folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Check if the result folder exists, if not, create it
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        # Build the full path of the log folder
        self.folder_path = os.path.join(result_folder, self.folder_name)
        # Create the log folder, if it already exists, do not raise an error
        os.makedirs(self.folder_path, exist_ok=True)
        # Build the full path of the log file
        self.log_file_path = os.path.join(self.folder_path, "log.jsonl")
        # Open the log file in write mode
        self.log_file = open(self.log_file_path, "w")
        # Create the README file and write the task description
        self.create_readme(task_description)

    def create_readme(self, task_description):
        """
        Create a README file in the log folder and write the task description into it.
        :param task_description: The description of the task to be written into the README file.
        """
        readme_path = os.path.join(self.folder_path, "README.md")
        with open(readme_path, "w") as readme_file:
            readme_file.write(f"# Task Description\n{task_description}")

    def log_iteration(self, data):
        """
        Write data in JSON format to the log file.
        :param data: The data to be written, which should be a JSON serializable object.
        """
        json.dump(data, self.log_file)
        self.log_file.write("\n")

    def save_file(self, file_name, content):
        """
        Save content to a file in the log folder.
        :param file_name: The name of the file to save.
        :param content: The content to be saved, should be in bytes.
        """
        file_path = os.path.join(self.folder_path, file_name)
        with open(file_path, "w") as f:
            json.dump(content, f, indent=4)

    def close(self):
        """
        Close the log file.
        """
        self.log_file.close()


def completion_finetune(model, messages):
    # Call the OpenAI API to generate completion
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    # Get the content of the first choice from the completion result
    return completion.choices[0].message.content