import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
import json
import random
import logging
from typing import List, Dict
import argparse
from model import Aligner, AlignerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)


class AlignmentDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length=512, max_evidence_count=4):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_evidence_count = max_evidence_count
        
        self.special_token = "[SPLIT]"
        if self.special_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.special_token])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        all_evidence = [(evi, label) for evi, label in zip(item['evidence'], item["annotation"])]
        scu_sentence = item['claim']
        
        if len(all_evidence) > 0:
            start_idx = random.randint(0, max(0, len(all_evidence) - 1))
            num_evidence = random.randint(1, self.max_evidence_count)
            sampled_evidence = all_evidence[start_idx:start_idx + num_evidence]
        else:
            sampled_evidence = []
        
        context_sentences, labels = zip(*sampled_evidence) if sampled_evidence else ([], [])
        
        text_input = scu_sentence + f" {self.special_token} " + f" {self.special_token} ".join(context_sentences)
        
        encoding = self.tokenizer(
            text_input,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        special_token_id = self.tokenizer.convert_tokens_to_ids(self.special_token)
        special_token_positions = (encoding['input_ids'] == special_token_id).nonzero(as_tuple=True)[1].tolist()
        
        special_token_positions = special_token_positions[:self.max_evidence_count]
        special_token_positions_padded = special_token_positions + [0] * (self.max_evidence_count - len(special_token_positions))
        
        labels_padded = list(labels) + [-1] * (self.max_evidence_count - len(labels))
        labels_mask = [1] * len(labels) + [0] * (self.max_evidence_count - len(labels))
        
        return {
            "input_ids": encoding['input_ids'].squeeze(0),
            "attention_mask": encoding['attention_mask'].squeeze(0),
            "special_token_positions": torch.tensor(special_token_positions_padded, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.float32),
            "labels_mask": torch.tensor(labels_mask, dtype=torch.int64)
        }



class AlignmentDataCollator:
    def __call__(self, features):

        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        special_token_positions = [f["special_token_positions"] for f in features]
        labels = torch.stack([f["labels"] for f in features])
        labels_mask = torch.stack([f["labels_mask"] for f in features])

        max_special_token_len = max(len(pos) for pos in special_token_positions)
        special_token_positions_padded = torch.zeros((len(features), max_special_token_len), dtype=torch.long)
        
        for i, pos in enumerate(special_token_positions):
            # special_token_positions_padded[i, :len(pos)] = torch.tensor(pos, dtype=torch.long)
            special_token_positions_padded[i, :len(pos)] = pos.clone().detach()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "special_token_positions": special_token_positions_padded,
            "labels": labels,
            "labels_mask": labels_mask
        }


def main(args):
    train_dataset_path = args.train_dataset
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    train_dataset = AlignmentDataset(file_path=train_dataset_path, tokenizer=tokenizer, max_evidence_count=args.max_evi)
    data_collator = AlignmentDataCollator()

    config = AlignerConfig(dropout=0.2)
    model = Aligner(config, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./sentence_alignment/results-model/",
        num_train_epochs=args.train_epoch,
        per_device_train_batch_size=8,
        logging_dir='./sentence_alignment/logs-model',
        logging_steps=100,
        save_strategy='epoch',
        learning_rate=2e-5,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    try:
        trainer.train()
    except RuntimeError as e:
        print(f"RuntimeError during training: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--train_dataset", type=str, default="dataset/train.json")
    parser.add_argument("--max_evi", type=int, default=8)
    parser.add_argument("--train_epoch", type=int, default=5)
    args = parser.parse_args()
    main(args)
