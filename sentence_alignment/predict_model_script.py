import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import logging
from typing import List, Dict
import argparse
from model import Aligner, AlignerConfig
from safetensors.torch import load_file
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_claim(model, tokenizer, claim, evidence_list, max_evidence_count=4, max_length=512):
    """
    Predicts a single claim and its associated pieces of evidence.
    
    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        claim: The claim sentence.
        evidence_list: List of evidence sentences.
        max_evidence_count: Number of evidence pieces processed at a time.
        max_length: Maximum input length.
    
    Returns:
        List of predictions, matching the length of evidence_list.
    """
    special_token = "[SPLIT]"
    predictions = [0] * len(evidence_list)  # Default all labels to 0
    
    # Process evidence in groups of max_evidence_count
    for i in range(0, len(evidence_list), max_evidence_count):
        chunk_evidence = evidence_list[i:i+max_evidence_count]
        
        # Construct input text
        text_input = claim + f" {special_token} " + f" {special_token} ".join(chunk_evidence)
        
        # Tokenize
        encoding = tokenizer(
            text_input,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Find positions of special tokens
        special_token_id = tokenizer.convert_tokens_to_ids(special_token)
        special_token_positions = (encoding['input_ids'] == special_token_id).nonzero(as_tuple=True)[1].tolist()
        
        # Ensure special token positions do not exceed evidence count
        special_token_positions = special_token_positions[:len(chunk_evidence)]
        special_token_positions_padded = special_token_positions + [0] * (max_evidence_count - len(special_token_positions))
        
        # Convert to tensor and move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        special_token_positions_tensor = torch.tensor([special_token_positions_padded], dtype=torch.long).to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_token_positions=special_token_positions_tensor
            )
            
            batch_predictions = (outputs['logits'] > 0).float().cpu().numpy()[0]
        
        # Update predictions
        for j in range(len(chunk_evidence)):
            if j < len(batch_predictions):
                pred_label = int(batch_predictions[j])
                predictions[i + j] = pred_label
    
    return predictions

def find_model_file(model_dir):
    """Finds the model file, supporting both .bin and .safetensors formats."""
    if os.path.isdir(model_dir):
        bin_file = os.path.join(model_dir, "pytorch_model.bin")
        safetensors_file = os.path.join(model_dir, "model.safetensors")
        
        if os.path.exists(safetensors_file):
            return safetensors_file, True
        elif os.path.exists(bin_file):
            return bin_file, False
        
        # Search for the latest checkpoint directory
        checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
            checkpoint_dir = os.path.join(model_dir, latest_checkpoint)
            
            bin_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
            safetensors_file = os.path.join(checkpoint_dir, "model.safetensors")
            
            if os.path.exists(safetensors_file):
                return safetensors_file, True
            elif os.path.exists(bin_file):
                return bin_file, False
    else:
        if model_dir.endswith(".safetensors"):
            return model_dir, True
        else:
            return model_dir, False
    
    raise FileNotFoundError(f"No model file found in {model_dir}")

def predict(model_dir, test_data_path, output_path, max_evidence_count=4, do_eval=False):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    special_token = "[SPLIT]"
    if special_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([special_token])
    
    config = AlignerConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = Aligner.from_pretrained(
        model_dir,
        config=config,
        tokenizer=tokenizer,
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    final_results = []
    eval_pred = []
    eval_true = []
    
    for idx, item in tqdm(enumerate(test_data), desc="Inference on eval set"):
        claim = item['claim']
        evidence_list = item['evidence']
        
        predictions = predict_single_claim(
            model=model,
            tokenizer=tokenizer,
            claim=claim,
            evidence_list=evidence_list,
            max_evidence_count=max_evidence_count
        )
        
        result_item = item.copy()
        result_item['prediction'] = predictions
        if do_eval:
            eval_pred.extend(predictions)
            eval_true.extend(item['annotation'])
        final_results.extend([result_item])
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    if do_eval:
        reports = classification_report(eval_true, eval_pred)
        matrix = confusion_matrix(eval_true, eval_pred)
        print(reports)
        print(matrix)
    
    logging.info(f"Prediction completed. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Prediction script")
    parser.add_argument("--model_dir", type=str, default="sentence_alignment/results-model/checkpoint-7500/")
    parser.add_argument("--test_data", type=str, default="dataset/test.json")
    parser.add_argument("--output_file", type=str, default="test_alignment.json")
    parser.add_argument("--max_evidence_count", type=int, default=8)
    parser.add_argument("--do_eval", action="store_true", default=False)
    args = parser.parse_args()
    
    predict(
        model_dir=args.model_dir,
        test_data_path=args.test_data,
        output_path=args.output_file,
        max_evidence_count=args.max_evidence_count,
        do_eval=args.do_eval
    )

if __name__ == '__main__':
    main()