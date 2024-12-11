import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from tqdm import tqdm

DATA_DIR = '/home/project/wasm-type-prediction/data'

def load_data(split='test'):
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

def evaluate_model(model_path="./type-prediction-model/checkpoint-5000"):
    print("Loading model for evaluation...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    model.eval()  # Set model to evaluation mode

    wasm_sequences, true_types = load_data('test')

    max_prompt_length = 900
    max_new_tokens = 5  # fewer tokens for faster generation
    num_beams = 1       # no beam search for speed
    num_return_sequences = 1

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    print("Starting evaluation on the full test set...")
    with torch.no_grad():  # no gradient calculation
        for wasm, true_type in tqdm(zip(wasm_sequences, true_types), total=len(wasm_sequences), desc="Evaluating"):
            input_text = f"Predict type for WebAssembly code:\n{wasm}\nType:"

            # Tokenize and truncate prompt if needed
            prompt_tokens = tokenizer(input_text, add_special_tokens=False)
            if len(prompt_tokens["input_ids"]) > max_prompt_length:
                prompt_tokens["input_ids"] = prompt_tokens["input_ids"][:max_prompt_length]
                prompt_tokens["attention_mask"] = [1]*len(prompt_tokens["input_ids"])

            input_ids = torch.tensor([prompt_tokens["input_ids"]], device=model.device)
            attention_mask = torch.tensor([prompt_tokens["attention_mask"]], device=model.device)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

            # Decode predictions
            predictions = []
            for output in outputs:
                pred = tokenizer.decode(output, skip_special_tokens=True)
                if "Type:" in pred:
                    pred = pred.split("Type:", 1)[1].strip()
                pred = pred.split("\n")[0].strip()
                predictions.append(pred)

            if predictions:
                # With num_return_sequences=1, predictions[0] is best guess
                if predictions[0] == true_type:
                    correct_top1 += 1
                # top-5 no longer matters much since num_return_sequences=1, 
                # but if you revert to more sequences, handle as before
                if true_type in predictions:
                    correct_top5 += 1
            total += 1

    print("\nFinal Results on full test set:")
    print(f"Top-1 Accuracy: {(correct_top1/total)*100:.2f}%")
    # Since now num_return_sequences=1, top-5 is essentially top-1
    print(f"Top-5 Accuracy: {(correct_top5/total)*100:.2f}%")

if __name__ == "__main__":
    # Run with nohup to pipe output to a file:
    # nohup python finetune.py > evaluation_output.log 2>&1 &
    evaluate_model()
