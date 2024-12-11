import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_from_disk
import sys
from tqdm import tqdm

DATA_DIR = '/home/project/wasm-type-prediction/data'
MAX_LENGTH = 512
TRAIN_TOKENIZED_PATH = "./tokenized_train"
TEST_TOKENIZED_PATH = "./tokenized_test"

def load_data(split='train'):
    """Load data from the data directory"""
    wasm_file = os.path.join(DATA_DIR, split, 'wasm.txt')
    type_file = os.path.join(DATA_DIR, split, 'type.txt')
    
    print(f"Loading {split} data...")
    with open(wasm_file, 'r') as f:
        wasm_sequences = [l.strip() for l in f]
    with open(type_file, 'r') as f:
        type_sequences = [l.strip() for l in f]
    
    if len(wasm_sequences) != len(type_sequences):
        raise ValueError("Mismatch between number of wasm and type lines.")
    
    print(f"Loaded {len(wasm_sequences)} {split} examples")
    return wasm_sequences, type_sequences

def prepare_dataset(tokenizer, split='train'):
    """Prepare dataset from wasm and type files"""
    tokenized_path = TRAIN_TOKENIZED_PATH if split == 'train' else TEST_TOKENIZED_PATH
    if os.path.exists(tokenized_path):
        print(f"Loading pre-tokenized {split} dataset from {tokenized_path}...")
        return load_from_disk(tokenized_path)
    
    wasm_sequences, type_sequences = load_data(split)
    combined = []
    for wasm, typ in zip(wasm_sequences, type_sequences):
        prompt = f"Predict type for WebAssembly code:\n{wasm}\nType:"
        completion = f" {typ}\n"
        combined.append((prompt, completion))

    def tokenize_function(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        tokenized = tokenizer(
            full_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )
        
        prompt_tokenized = tokenizer(
            prompts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )
        
        prompt_lengths = (prompt_tokenized["input_ids"] != tokenizer.pad_token_id).sum(axis=1)
        labels = tokenized["input_ids"].copy()
        for i, pl in enumerate(prompt_lengths):
            pl_int = int(pl)
            labels[i, :pl_int] = -100
        tokenized["labels"] = labels
        return tokenized

    dataset = Dataset.from_dict({
        "prompt": [p for p, c in combined],
        "completion": [c for p, c in combined]
    })
    
    print(f"Tokenizing {split} dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        desc=f"Tokenizing {split} dataset",
        remove_columns=dataset.column_names
    )
    
    tokenized_dataset.save_to_disk(tokenized_path)
    return tokenized_dataset

def train_model():
    print("Loading model and tokenizer...")
    # Use a smaller model
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        truncation_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Preparing datasets...")
    train_dataset = prepare_dataset(tokenizer, 'train')
    eval_dataset = prepare_dataset(tokenizer, 'test')
    
    num_epochs = 3
    batch_size = 10  # smaller batch due to memory constraints
    gradient_accumulation_steps = 8
    total_steps = (len(train_dataset) * num_epochs) // (batch_size * gradient_accumulation_steps)
    
    training_args = TrainingArguments(
        output_dir="./type-prediction-model",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=total_steps // 10 if total_steps > 10 else 0,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        gradient_checkpointing=False,  # disabled to avoid issues
        save_total_limit=2,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    try:
        trainer.train(resume_from_checkpoint="./type-prediction-model/checkpoint-5000")
    except Exception as e:
        print(f"Training interrupted: {e}")

    
    print("Saving final model...")
    trainer.save_model("./final-type-prediction-model")

def evaluate_model(model_path="./type-prediction-model/checkpoint-5000"):
    print("Loading model for evaluation...")
    # Load model from the checkpoint
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Load tokenizer from the original model name used during training
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    wasm_sequences, true_types = load_data('test')

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    print("Starting evaluation...")
    for wasm, true_type in tqdm(zip(wasm_sequences, true_types), total=len(wasm_sequences), desc="Evaluating"):
        input_text = f"Predict type for WebAssembly code:\n{wasm}\nType:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=5,
            num_return_sequences=2,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7
        )

        predictions = []
        for output in outputs:
            pred = tokenizer.decode(output, skip_special_tokens=True)
            if "Type:" in pred:
                pred = pred.split("Type:")[1].strip()
            pred = pred.split("\n")[0].strip()
            predictions.append(pred)

        if len(predictions) > 0:
            if predictions[0] == true_type:
                correct_top1 += 1
            if true_type in predictions:
                correct_top5 += 1
        total += 1

    print("\nFinal Results:")
    print(f"Top-1 Accuracy: {(correct_top1/total)*100:.2f}%")
    print(f"Top-5 Accuracy: {(correct_top5/total)*100:.2f}%")
# def evaluate_model(model_path="./type-prediction-model/checkpoint-5000", num_samples=200):
#     print("Loading model for evaluation...")
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
#     wasm_sequences, true_types = load_data('test')

#     # Take a subset of the data
#     # For a random sample, you can also use `random.sample`:
#     # import random
#     # indices = random.sample(range(len(wasm_sequences)), num_samples)
#     # wasm_sequences = [wasm_sequences[i] for i in indices]
#     # true_types = [true_types[i] for i in indices]

#     # Or just take the first num_samples
#     wasm_sequences = wasm_sequences[:num_samples]
#     true_types = true_types[:num_samples]

#     correct_top1 = 0
#     correct_top5 = 0
#     total = 0
    
#     print("Starting evaluation on a subset of", num_samples, "samples...")
#     for wasm, true_type in tqdm(zip(wasm_sequences, true_types), total=len(wasm_sequences), desc="Evaluating"):
#         input_text = f"Predict type for WebAssembly code:\n{wasm}\nType:"
#         inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)
        
#         outputs = model.generate(
#             inputs["input_ids"],
#             max_new_tokens=20,
#             num_return_sequences=5,
#             num_beams=5,
#             early_stopping=True,
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=False
#         )
        
#         predictions = []
#         for output in outputs:
#             pred = tokenizer.decode(output, skip_special_tokens=True)
#             if "Type:" in pred:
#                 pred = pred.split("Type:")[1].strip()
#             pred = pred.split("\n")[0].strip()
#             predictions.append(pred)
        
#         if len(predictions) > 0:
#             if predictions[0] == true_type:
#                 correct_top1 += 1
#             if true_type in predictions:
#                 correct_top5 += 1
#         total += 1
        
#     print("\nFinal Results on subset:")
#     print(f"Top-1 Accuracy: {(correct_top1/total)*100:.2f}%")
#     print(f"Top-5 Accuracy: {(correct_top5/total)*100:.2f}%")

# Then call:
# evaluate_model(num_samples=1000)

if __name__ == "__main__":
    train_model()
    # evaluate_model()
