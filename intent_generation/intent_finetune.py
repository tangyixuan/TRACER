import json
from tqdm import tqdm
from utils.utils import *
from intent_generation.prompts import *

def finetune_gpt_mini(datapath, result_folder):
    # TODO: Prepare finetune data. Finetune GPT-4o-mini API
    # data example
    # {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
    
    task_description = """This task is to generate the finetune data for intent generation."""

    result_handler = DataHandler(task_description, result_folder)

    with open(datapath, 'r') as f:
        data = json.load(f)
    for event in data:
        evidence = []
        annotation = []
        for evi, label in zip(event['evidence'], event['annotation']):
            if label != 0:
                evidence.append(evi)
                annotation.append(label)
        event['evidence'] = evidence
        event['annotation'] = annotation
    
    messages = []
    for event in tqdm(data):
        if event["intent_valid"] == 0:
            continue
        claim = event['claim']
        evidence = '\n'.join(event['evidence'])
        inputs = FINETUNE_TEMPLATE_TASK.replace("{claim}", claim).replace("{evidence}", evidence)
        outputs = FINETUNE_TEMPLATE_OUTPUT.format(intent=event['intent_gold'])
        if isinstance(outputs, list):
            print(event['example_id'])
        message = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": inputs
                },
                {
                    "role": "assistant",
                    "content": outputs
                }
            ]
        }
        messages.append(message)
    
    with open(f'{result_handler.folder_path}/finetune_data.jsonl', 'w') as f:
        for message in messages:
            json.dump(message, f)
            f.write('\n')

    result_handler.close()


finetune_gpt_mini("dataset/train.json", "intent_generation/finetune_data/")