import argparse
import torch
from models.basemodel import Network, NetworkMargin
import utils.config as config
from utils.dataset import WaterBirds, CelebaDataset
from torch.utils.data import DataLoader
from utils.utils import compute_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Run predictions with a trained model.')
    parser.add_argument('--type', type=str, required=True, choices=['baseline', 'margin'],
                        help='Type of model to use: baseline (ERM) or margin (CAML).')
    parser.add_argument('--technique', type=str, required=True, choices=['erm', 'ema', 'swa'],
                        help='Training technique used for the model.')
    parser.add_argument('--dataset', type=str, default='waterbirds', choices=['waterbirds', 'celeba'],
                        help='Dataset to use for prediction.')
    parser.add_argument("--gpu", type=str, default='0', help='GPU card ID.')
    return parser.parse_args()

def get_model_path(model_type, technique):
    """Selects the correct model path from config based on type and technique."""
    if model_type == 'baseline':
        if technique == 'erm':
            return config.baseline_path_erm
        elif technique == 'ema':
            return config.baseline_path_ema
        elif technique == 'swa':
            return config.baseline_path_swa
    elif model_type == 'margin':
        if technique == 'erm':
            return config.margin_path_erm
        elif technique == 'ema':
            return config.margin_path_ema
        elif technique == 'swa':
            return config.margin_path_swa
    return None

def predict(args):
    """
    Loads a trained model and runs predictions on the test set.
    """
    DEVICE = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 1. Define the model architecture
    if args.type == 'baseline':
        model = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim)
    else: # margin
        model = NetworkMargin(config.model_name, config.num_class, DEVICE, config.std, config.mlp_neurons, config.hid_dim)
    model.to(DEVICE)

    # 2. Load the saved weights
    model_path = get_model_path(args.type, args.technique)
    if model_path is None:
        print(f"Error: Could not determine model path for type '{args.type}' and technique '{args.technique}'.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first.")
        return

    # 3. Set the model to evaluation mode
    model.eval()

    # 4. Load the test dataset
    if args.dataset == 'waterbirds':
        test_dataset = WaterBirds(split='test')
    else: # celeba
        test_dataset = CelebaDataset(split=2)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=config.base_batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # 5. Run evaluation
    print(f"\nRunning evaluation on the {args.dataset} test set...")
    with torch.no_grad():
        test_acc, test_worst, test_avg = compute_accuracy(model, test_loader, DEVICE)
        print(f"\nModel: {args.type} ({args.technique})")
        print(f"Global Accuracy: {test_acc:.2f}%")
        print(f"Worst-Group Accuracy: {test_worst*100:.2f}%")
        print(f"Average-Group Accuracy: {test_avg*100:.2f}%")

if __name__ == '__main__':
    args = parse_args()
    predict(args)