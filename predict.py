import argparse
import torch
import os
from models.basemodel import Network, NetworkMargin
import utils.config as config
from utils.dataset import WaterBirds, CelebaDataset
from torch.utils.data import DataLoader
from utils.utils import compute_accuracy

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Run and compare predictions for baseline and CAML models.')
    parser.add_argument('--technique', type=str, required=True, choices=['erm', 'ema', 'swa'],
                        help='Training technique to evaluate (ERM, EMA, or SWA).')
    parser.add_argument('--dataset', type=str, default='waterbirds', choices=['waterbirds', 'celeba'],
                        help='Dataset to use for prediction.')
    parser.add_argument("--gpu", type=str, default='0', help='GPU card ID.')
    return parser.parse_args()

def get_model_paths(technique):
    """Selects the correct model paths from config based on the technique."""
    paths = {}
    if technique == 'erm':
        paths['baseline'] = config.baseline_path_erm
        paths['margin'] = config.margin_path_erm
    elif technique == 'ema':
        paths['baseline'] = config.baseline_path_ema
        paths['margin'] = config.margin_path_ema
    elif technique == 'swa':
        paths['baseline'] = config.baseline_path_swa
        paths['margin'] = config.margin_path_swa
    return paths

def run_evaluation(model, model_path, test_loader, device):
    """
    Helper function to load model weights and compute accuracy.
    Returns (None, None, None) if the model file is not found.
    """
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at '{model_path}'. Skipping evaluation for this model.")
        return None, None, None
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
        model.eval()
        with torch.no_grad():
            test_acc, test_worst, test_avg = compute_accuracy(model, test_loader, device)
        return test_acc, test_worst, test_avg
    except Exception as e:
        print(f"An error occurred while evaluating model {model_path}: {e}")
        return None, None, None

def predict_comparison(args):
    """
    Loads baseline and CAML models for a given technique, runs predictions,
    and prints a comparison of their performance.
    """
    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        # 1. Get paths for the models to compare

        # The baseline is always the standard ERM model for a consistent benchmark.

    baseline_model_path = config.baseline_path_erm

    caml_model_path = get_model_paths(args.technique)['margin'] # Get margin model for the specified technique

    

        # 2. Load the test dataset

    if args.dataset == 'waterbirds':

        test_dataset = WaterBirds(split='test')

    else: # celeba

        test_dataset = CelebaDataset(split=2)

    

    test_loader = DataLoader(dataset=test_dataset,

                                batch_size=config.base_batch_size,

                                shuffle=False,

                                num_workers=4,

                                pin_memory=True)

        

    print(f"\nRunning comparison for technique '{args.technique.upper()}' on dataset '{args.dataset}'...")

    

        # 3. Evaluate Baseline Model (ERM)

    print("\n--- Evaluating Baseline Model (ERM) ---")

    baseline_model = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim).to(DEVICE)

    base_results = run_evaluation(baseline_model, baseline_model_path, test_loader, DEVICE)

    

        # 4. Evaluate Margin Model (CAML + Technique)

    print(f"\n--- Evaluating CAML Model ({args.technique.upper()}) ---")

    margin_model = NetworkMargin(config.model_name, config.num_class, DEVICE, config.std, config.mlp_neurons, config.hid_dim).to(DEVICE)

    margin_results = run_evaluation(margin_model, caml_model_path, test_loader, DEVICE)

    

        # 5. Print comparison table

    print("\n" + "="*65)

    print("                 Prediction Results Comparison")

    print("="*65)

    print(f"Technique: CAML + {args.technique.upper()} | Dataset: {args.dataset}")

    print("-" * 65)

    print(f"{'Metric':<28} | {'ERM Baseline':<17} | {'CAML Model':<17}")

    print("-" * 65)

    metric_names = ['Global Accuracy (%)', 'Worst-Group Accuracy (%)', 'Average-Group Accuracy (%)']
    
    base_acc, base_worst, base_avg = base_results
    margin_acc, margin_worst, margin_avg = margin_results

    # Format results for printing
    base_formatted = [f"{base_acc:.2f}", f"{base_worst*100:.2f}", f"{base_avg*100:.2f}"] if all(r is not None for r in base_results) else ["N/A"] * 3
    margin_formatted = [f"{margin_acc:.2f}", f"{margin_worst*100:.2f}", f"{margin_avg*100:.2f}"] if all(r is not None for r in margin_results) else ["N/A"] * 3

    for i, name in enumerate(metric_names):
        print(f"{name:<28} | {base_formatted[i]:<15} | {margin_formatted[i]:<15}")

    if all(r is None for r in base_results) and all(r is None for r in margin_results):
        print("\nNo trained models found for this technique. Please run training first.")

    print("="*60)


if __name__ == '__main__':
    args = parse_args()
    predict_comparison(args)
