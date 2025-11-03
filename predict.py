
import torch
from models.basemodel import Network
import utils.config as config
from utils.dataset import WaterBirds
from torch.utils.data import DataLoader
from utils.utils import compute_accuracy

def predict_with_ema_model():
    """
    Loads the trained EMA model and runs predictions on the test set.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Define the model architecture (must be same as training)
    model = Network(config.model_name, config.num_class, config.mlp_neurons, config.hid_dim)
    model.to(DEVICE)

    # 2. Load the saved weights
    model_path = config.basemodel_path
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first and the path in utils/config.py is correct.")
        return

    # 3. Set the model to evaluation mode
    model.eval()

    # Load the test dataset
    test_dataset = WaterBirds(split='test')
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=config.base_batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)

    # Run evaluation
    print("\nRunning evaluation on the test set...")
    with torch.no_grad():
        test_acc, test_worst, test_avg = compute_accuracy(model, test_loader, DEVICE)
        print(f"\nGlobal Accuracy: {test_acc:.2f}%")
        print(f"Worst-Group Accuracy: {test_worst*100:.2f}%")
        print(f"Average-Group Accuracy: {test_avg*100:.2f}%")

if __name__ == '__main__':
    predict_with_ema_model()
