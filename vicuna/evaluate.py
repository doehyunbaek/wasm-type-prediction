import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from datetime import datetime
import sys

# Constants
DATA_DIR = '/home/project/wasm-type-prediction/data'
MAX_SEQ_LENGTH = 1024 - 5  # Reserve space for generated tokens

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(log_file):
    """Setup logging to both file and console with tqdm compatibility"""
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)

    # Setup console handler with tqdm compatibility
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(console_formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger

def load_data(split='test'):
    """Load data from the data directory"""
    wasm_file = os.path.join(DATA_DIR, split, 'wasm.txt')
    type_file = os.path.join(DATA_DIR, split, 'type.txt')
    
    try:
        # Check if files exist
        if not os.path.exists(wasm_file):
            raise FileNotFoundError(f"WASM file not found: {wasm_file}")
        if not os.path.exists(type_file):
            raise FileNotFoundError(f"Type file not found: {type_file}")
            
        with open(wasm_file, 'r', encoding='utf-8') as f:
            wasm_sequences = [line.strip() for line in f if line.strip()]
        
        with open(type_file, 'r', encoding='utf-8') as f:
            type_sequences = [line.strip() for line in f if line.strip()]
        
        if len(wasm_sequences) != len(type_sequences):
            raise ValueError(f"Mismatch between number of WASM sequences ({len(wasm_sequences)}) "
                           f"and type sequences ({len(type_sequences)})")
        
        return wasm_sequences, type_sequences
        
    except Exception as e:
        raise RuntimeError(f"Error loading {split} data: {str(e)}")

class ProgressCallback:
    def __init__(self, total, logger):
        self.pbar = tqdm(total=total, desc="Evaluating")
        self.logger = logger
        self.last_log = 0
        self.log_interval = 100  # Log progress every 100 steps

    def update(self, n=1):
        self.pbar.update(n)
        current = self.pbar.n
        if current - self.last_log >= self.log_interval:
            progress = (current / self.pbar.total) * 100
            self.logger.info(f"Progress: {progress:.2f}% ({current}/{self.pbar.total})")
            self.last_log = current

    def close(self):
        self.pbar.close()

def evaluate_model(
    model_path,
    batch_size=1,
    num_return_sequences=5,
    temperature=0.7,
    sample_interval=1000
):
    """
    Evaluate the model on test data with enhanced progress logging
    """
    # Setup logging
    log_file = f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logging(log_file)
    
    logger.info("Starting evaluation process...")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Output log file: {log_file}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load test data
        logger.info("Loading test data...")
        wasm_sequences, true_types = load_data('test')
        logger.info(f"Loaded {len(wasm_sequences)} test examples")
        
        # Initialize metrics
        metrics = {
            "correct_top1": 0,
            "correct_top5": 0,
            "total": 0,
            "errors": 0,
            "sequence_length_issues": 0
        }
        
        # Create progress callback
        progress = ProgressCallback(len(wasm_sequences), logger)
        
        # Create results directory if it doesn't exist
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Open detailed results file
        detailed_results_file = os.path.join(results_dir, f'detailed_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(detailed_results_file, 'w', encoding='utf-8') as results_f:
            # Process examples
            logger.info("Starting evaluation...")
            for idx in range(0, len(wasm_sequences), batch_size):
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
                            if pred:
                                predictions.append(pred)
                        
                        # Update metrics
                        if predictions:
                            if predictions[0] == true_type:
                                metrics["correct_top1"] += 1
                            if true_type in predictions:
                                metrics["correct_top5"] += 1
                        
                        metrics["total"] += 1
                        
                        # Write detailed results
                        results_f.write(f"Example {current_idx}:\n")
                        results_f.write(f"True type: {true_type}\n")
                        results_f.write(f"Predictions: {predictions[:5]}\n")
                        results_f.write("-" * 50 + "\n")
                        results_f.flush()  # Ensure writing to file
                        
                        # Log sample predictions
                        if current_idx % sample_interval == 0:
                            logger.info(f"\nSample prediction {current_idx}:")
                            logger.info(f"True type: {true_type}")
                            logger.info(f"Predicted types: {predictions[:5]}")
                            logger.info(f"Current Top-1 Accuracy: {(metrics['correct_top1']/metrics['total'])*100:.2f}%")
                        
                    except Exception as e:
                        metrics["errors"] += 1
                        logger.error(f"Error processing example {current_idx}: {str(e)}")
                        continue
                    
                    # Update progress
                    progress.update()
        
        # Close progress bar
        progress.close()
        
        # Log final results
        logger.info("\nEvaluation Results:")
        logger.info(f"Total examples processed: {metrics['total']}")
        logger.info(f"Top-1 Accuracy: {(metrics['correct_top1']/metrics['total'])*100:.2f}%")
        logger.info(f"Top-5 Accuracy: {(metrics['correct_top5']/metrics['total'])*100:.2f}%")
        logger.info(f"Number of errors: {metrics['errors']}")
        logger.info(f"Sequences requiring truncation: {metrics['sequence_length_issues']}")
        logger.info(f"Detailed results saved to: {detailed_results_file}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Fatal error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate WASM Type Prediction Model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for evaluation')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Temperature for generation sampling')
    parser.add_argument('--sample_interval', type=int, default=1000,
                      help='How often to print sample predictions')
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        metrics = evaluate_model(
            model_path=args.model_path,
            batch_size=args.batch_size,
            temperature=args.temperature,
            sample_interval=args.sample_interval
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        sys.exit(1)