import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

# Constants
DATA_DIR = '/home/project/wasm-type-prediction/data'
MODEL_NAME = "distilgpt2"
MAX_SEQ_LENGTH = 1024 - 5  # Reserve space for generated tokens

def load_data(split='test'):
    """Load data from the data directory"""
    wasm_file = os.path.join(DATA_DIR, split, 'wasm.txt')
    type_file = os.path.join(DATA_DIR, split, 'type.txt')
    
    logging.info(f"Loading {split} data...")
    
    try:
        with open(wasm_file, 'r') as f:
            wasm_sequences = [l.strip() for l in f]
        with open(type_file, 'r') as f:
            type_sequences = [l.strip() for l in f]
        
        if len(wasm_sequences) != len(type_sequences):
            raise ValueError("Mismatch between number of wasm and type lines.")
        
        logging.info(f"Loaded {len(wasm_sequences)} {split} examples")
        return wasm_sequences, type_sequences
    
    except FileNotFoundError as e:
        logging.error(f"Could not find data files: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def evaluate_model(
    model_path="./type-prediction-model/checkpoint-5000",
    batch_size=1,
    num_return_sequences=5,
    temperature=0.7,
    sample_interval=1000
):
    """
    Evaluate the model on test data
    
    Args:
        model_path (str): Path to the trained model
        batch_size (int): Batch size for evaluation
        num_return_sequences (int): Number of predictions to generate per input
        temperature (float): Temperature for generation sampling
        sample_interval (int): How often to print sample predictions
    """
    logging.info("Starting evaluation process...")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Batch size: {batch_size}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Load model and tokenizer
        logging.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load test data
        wasm_sequences, true_types = load_data('test')
        
        # Initialize metrics
        metrics = {
            "correct_top1": 0,
            "correct_top5": 0,
            "total": 0,
            "errors": 0,
            "sequence_length_issues": 0
        }
        
        # Create detailed results list
        detailed_results = []
        
        # Process examples
        logging.info("Starting evaluation...")
        for idx in tqdm(range(0, len(wasm_sequences), batch_size), desc="Evaluating"):
            batch_end = min(idx + batch_size, len(wasm_sequences))
            batch_wasm = wasm_sequences[idx:batch_end]
            batch_true = true_types[idx:batch_end]
            
            for sample_idx, (wasm, true_type) in enumerate(zip(batch_wasm, batch_true)):
                current_idx = idx + sample_idx
                
                try:
                    # Prepare input
                    input_text = f"Predict type for WebAssembly code:\n{wasm}\nType:"
                    
                    # Tokenize with length limitation
                    inputs = tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_SEQ_LENGTH,
                        padding=False
                    ).to(device)
                    
                    # Check sequence length
                    if inputs["input_ids"].shape[1] >= MAX_SEQ_LENGTH:
                        metrics["sequence_length_issues"] += 1
                        logging.warning(f"Example {current_idx} truncated from {inputs['input_ids'].shape[1]} tokens")
                    
                    # Generate predictions
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=5,
                            num_return_sequences=num_return_sequences,
                            num_beams=5,
                            early_stopping=True,
                            pad_token_id=tokenizer.eos_token_id,
                            temperature=temperature,
                            do_sample=True
                        )
                    
                    # Process predictions
                    predictions = []
                    for output in outputs:
                        pred = tokenizer.decode(output, skip_special_tokens=True)
                        if "Type:" in pred:
                            pred = pred.split("Type:")[1].strip()
                        pred = pred.split("\n")[0].strip()
                        if pred:  # Only add non-empty predictions
                            predictions.append(pred)
                    
                    # Update metrics
                    if predictions:
                        if predictions[0] == true_type:
                            metrics["correct_top1"] += 1
                        if true_type in predictions:
                            metrics["correct_top5"] += 1
                    
                    metrics["total"] += 1
                    
                    # Store detailed results
                    detailed_results.append({
                        "index": current_idx,
                        "true_type": true_type,
                        "predictions": predictions[:5],
                        "correct_top1": predictions and predictions[0] == true_type,
                        "correct_top5": predictions and true_type in predictions
                    })
                    
                    # Print sample predictions
                    if current_idx % sample_interval == 0:
                        logging.info(f"\nSample prediction {current_idx}:")
                        logging.info(f"True type: {true_type}")
                        logging.info(f"Predicted types: {predictions[:5]}")
                    
                except Exception as e:
                    metrics["errors"] += 1
                    logging.error(f"Error processing example {current_idx}: {str(e)}")
                    continue
        
        # Calculate and log final results
        logging.info("\nEvaluation Results:")
        logging.info(f"Total examples processed: {metrics['total']}")
        logging.info(f"Top-1 Accuracy: {(metrics['correct_top1']/metrics['total'])*100:.2f}%")
        logging.info(f"Top-5 Accuracy: {(metrics['correct_top5']/metrics['total'])*100:.2f}%")
        logging.info(f"Number of errors: {metrics['errors']}")
        logging.info(f"Sequences requiring truncation: {metrics['sequence_length_issues']}")
        
        # Save detailed results
        results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            for result in detailed_results:
                f.write(f"Example {result['index']}:\n")
                f.write(f"True type: {result['true_type']}\n")
                f.write(f"Predictions: {result['predictions']}\n")
                f.write(f"Correct top-1: {result['correct_top1']}\n")
                f.write(f"Correct top-5: {result['correct_top5']}\n")
                f.write("-" * 50 + "\n")
        
        logging.info(f"Detailed results saved to {results_file}")
        
        return {
            "top1_accuracy": (metrics['correct_top1']/metrics['total'])*100,
            "top5_accuracy": (metrics['correct_top5']/metrics['total'])*100,
            "total_examples": metrics['total'],
            "errors": metrics['errors'],
            "truncated_sequences": metrics['sequence_length_issues']
        }
    
    except Exception as e:
        logging.error(f"Fatal error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate WASM Type Prediction Model')
    parser.add_argument('--model_path', type=str, default="./type-prediction-model/checkpoint-5000",
                      help='Path to the trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for evaluation')
    parser.add_argument('--num_return_sequences', type=int, default=5,
                      help='Number of predictions to generate per input')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for generation sampling')
    parser.add_argument('--sample_interval', type=int, default=1000,
                      help='How often to print sample predictions')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(
            model_path=args.model_path,
            batch_size=args.batch_size,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            sample_interval=args.sample_interval
        )
        
        print("\nEvaluation completed successfully!")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        exit(1)