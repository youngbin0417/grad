import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.config as config
from torch.utils.data import DataLoader
from utils.dataset import CelebaDataset, WaterBirds
from utils.utils import compute_accuracy
from models.basemodel import Network, NetworkMargin
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='celeba',
                        help='which dataset to evaluate on?')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu card ID')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size for evaluation')
    args = parser.parse_args()
    return args

def load_model(model_class, model_path, device, model_type='baseline', **kwargs):
    """Load a model from the specified path"""
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, device, model_type='baseline', margin_model=None):
    """Evaluate a single model"""
    model.eval()
    all_preds = []
    all_targets = []
    all_group_ids = []
    
    with torch.no_grad():
        for _, (_, features, targets, bias, _) in enumerate(data_loader):
            features = features.to(device)
            targets_device = targets.to(device)
            
            if model_type == 'baseline':
                logits, _, _ = model(features)
            else:  # margin or ensemble, which are handled similarly in eval
                # In eval mode, NetworkMargin returns 3 values (cosine, probas, features)
                # The first value (cosine) serves as the logits.
                logits, _, _ = model(features, m=None, s=config.scale)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Correctly calculate group_id
            group_ids = targets.numpy() * 2 + bias.numpy()
            all_group_ids.extend(group_ids)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_group_ids = np.array(all_group_ids)
    
    # Calculate metrics
    global_acc = (all_preds == all_targets).mean()
    
    # Calculate group-specific accuracies
    unique_groups = np.unique(all_group_ids)
    group_accs = []
    for group in unique_groups:
        group_mask = all_group_ids == group
        group_acc = (all_preds[group_mask] == all_targets[group_mask]).mean()
        group_accs.append(group_acc)
    
    worst_acc = min(group_accs) if group_accs else 0.0
    avg_acc = np.mean(group_accs) if group_accs else 0.0
    
    return global_acc, worst_acc, avg_acc, group_accs

def evaluate_ensemble(models, data_loader, device, model_type='baseline'):
    """Evaluate ensemble of models"""
    all_logits = []
    all_targets = []
    all_group_ids = []
    
    with torch.no_grad():
        for _, (_, features, targets, bias, _) in enumerate(data_loader):
            features = features.to(device)
            targets_device = targets.to(device)
            
            # Collect logits from all models in the ensemble
            batch_logits_list = []
            for model in models:
                if model_type == 'baseline':
                    logits, _, _ = model(features)
                else:  # margin
                    # In eval mode, NetworkMargin returns 3 values. The first is the logits (cosine).
                    logits, _, _ = model(features, m=None, s=config.scale)
                batch_logits_list.append(logits.unsqueeze(0))
            
            # Average logits across ensemble models
            avg_logits = torch.cat(batch_logits_list, dim=0).mean(dim=0)
            
            all_logits.append(avg_logits.cpu())
            all_targets.extend(targets.cpu().numpy())

            # Correctly calculate group_id
            group_ids = targets.numpy() * 2 + bias.numpy()
            all_group_ids.extend(group_ids)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = np.array(all_targets)
    all_group_ids = np.array(all_group_ids)
    all_preds = torch.argmax(all_logits, dim=1).numpy()
    
    # Calculate metrics
    global_acc = (all_preds == all_targets).mean()
    
    # Calculate group-specific accuracies
    unique_groups = np.unique(all_group_ids)
    group_accs = []
    for group in unique_groups:
        group_mask = all_group_ids == group
        group_acc = (all_preds[group_mask] == all_targets[group_mask]).mean()
        group_accs.append(group_acc)
    
    worst_acc = min(group_accs) if group_accs else 0.0
    avg_acc = np.mean(group_accs) if group_accs else 0.0
    
    return global_acc, worst_acc, avg_acc, group_accs

def compare_models(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create data loader
    batch_size = args.batch_size
    if args.dataset == 'celeba':
        test_dataset = CelebaDataset(split=2)  # split=2 is test
    elif args.dataset == 'waterbirds':
        test_dataset = WaterBirds(split='test')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    
    # Define model configurations
    baseline_kwargs = {
        'model_name': config.model_name,
        'num_classes': config.num_class,
        'mlp_neurons': config.mlp_neurons,
        'hid_dim': config.hid_dim
    }
    
    margin_kwargs = {
        'model_name': config.model_name,
        'num_classes': config.num_class,
        'DEVICE': device,
        'std': config.std,
        'mlp_neurons': config.mlp_neurons,
        'hid_dim': config.hid_dim
    }
    
    results = []
    
    # Load and evaluate baseline model
    if os.path.exists(config.baseline_path_erm):
        baseline_model = load_model(Network, config.baseline_path_erm, device, **baseline_kwargs)
        global_acc, worst_acc, avg_acc, group_accs = evaluate_model(
            baseline_model, test_loader, device, model_type='baseline')
        results.append(['Baseline', global_acc, worst_acc, avg_acc])
    
    # Load and evaluate EMA baseline model
    if os.path.exists(config.baseline_path_ema):
        ema_baseline_model = load_model(Network, config.baseline_path_ema, device, **baseline_kwargs)
        global_acc, worst_acc, avg_acc, group_accs = evaluate_model(
            ema_baseline_model, test_loader, device, model_type='baseline')
        results.append(['Baseline-EMA', global_acc, worst_acc, avg_acc])
    
    # Load and evaluate SWA baseline model
    if os.path.exists(config.baseline_path_swa):
        swa_baseline_model = load_model(Network, config.baseline_path_swa, device, **baseline_kwargs)
        global_acc, worst_acc, avg_acc, group_accs = evaluate_model(
            swa_baseline_model, test_loader, device, model_type='baseline')
        results.append(['Baseline-SWA', global_acc, worst_acc, avg_acc])
    
    # Load and evaluate margin model
    if os.path.exists(config.margin_path_erm):
        margin_model = load_model(NetworkMargin, config.margin_path_erm, device, **margin_kwargs)
        global_acc, worst_acc, avg_acc, group_accs = evaluate_model(
            margin_model, test_loader, device, model_type='margin')
        results.append(['Margin', global_acc, worst_acc, avg_acc])
    
    # Load and evaluate EMA margin model
    if os.path.exists(config.margin_path_ema):
        ema_margin_model = load_model(NetworkMargin, config.margin_path_ema, device, **margin_kwargs)
        global_acc, worst_acc, avg_acc, group_accs = evaluate_model(
            ema_margin_model, test_loader, device, model_type='margin')
        results.append(['Margin-EMA', global_acc, worst_acc, avg_acc])
    
    # Load and evaluate SWA margin model
    if os.path.exists(config.margin_path_swa):
        swa_margin_model = load_model(NetworkMargin, config.margin_path_swa, device, **margin_kwargs)
        global_acc, worst_acc, avg_acc, group_accs = evaluate_model(
            swa_margin_model, test_loader, device, model_type='margin')
        results.append(['Margin-SWA', global_acc, worst_acc, avg_acc])
    
    # Try to load and evaluate ensemble models using Top-K selection
    topk = 5
    metric = 'worst' # 'worst' or 'val'

    # --- Helper function for Top-K evaluation ---
    def evaluate_topk_ensemble(model_type, training_type, model_class, model_kwargs):
        print(f"\nAttempting to evaluate Top-K ensemble for: {model_type}-{training_type}")
        ensemble_paths = [f'ensemble_model/{training_type}_{model_type}_epoch{i}.pt' for i in range(config.base_epochs)]
        val_accs_path = f'ensemble_model/{training_type}_{model_type}_val_accs.npy'
        val_worsts_path = f'ensemble_model/{training_type}_{model_type}_val_worsts.npy'

        if not (os.path.exists(val_accs_path) and os.path.exists(val_worsts_path)):
            print(f"Validation score files not found. Skipping Top-K ensemble.")
            return

        val_accs = np.load(val_accs_path)
        val_worsts = np.load(val_worsts_path)
        
        metric_arr = val_worsts if metric == 'worst' else val_accs
        
        # Ensure we don't have more epochs in metric_arr than in existing paths
        num_epochs = min(len(metric_arr), len(ensemble_paths))
        metric_arr = metric_arr[:num_epochs]
        
        # Get indices of top-k models
        # Ensure topk is not larger than the number of available epochs
        actual_topk = min(topk, len(metric_arr))
        if actual_topk == 0:
            print("No epochs to select from. Skipping.")
            return
            
        topk_indices = np.argsort(metric_arr)[-actual_topk:]
        
        ensemble_models = []
        for idx in topk_indices:
            if idx < len(ensemble_paths):
                path = ensemble_paths[idx]
                if os.path.exists(path):
                    try:
                        model = load_model(model_class, path, device, **model_kwargs)
                        ensemble_models.append(model)
                    except Exception as e:
                        print(f"Could not load model {path}: {e}")
                        continue
        
        if ensemble_models:
            print(f"Evaluating with {len(ensemble_models)} models based on Top-K '{metric}' metric.")
            global_acc, worst_acc, avg_acc, group_accs = evaluate_ensemble(
                ensemble_models, test_loader, device, model_type=model_type)
            
            model_name = f'{model_type.capitalize()}-{training_type.upper()}-Ensemble (Top-{len(ensemble_models)})'
            results.append([model_name, global_acc, worst_acc, avg_acc])
        else:
            print("No models loaded for ensemble. Skipping.")

    # --- Evaluate all ensemble combinations ---
    evaluate_topk_ensemble('baseline', 'ema', Network, baseline_kwargs)
    evaluate_topk_ensemble('baseline', 'swa', Network, baseline_kwargs)
    evaluate_topk_ensemble('margin', 'ema', NetworkMargin, margin_kwargs)
    evaluate_topk_ensemble('margin', 'swa', NetworkMargin, margin_kwargs)
    
    # Create a comparison table
    df = pd.DataFrame(results, columns=['Model', 'Global Acc', 'Worst Acc', 'Avg Acc'])
    print("\nModel Comparison Results:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Optionally save results to CSV
    df.to_csv(f'{args.dataset}_comparison_results.csv', index=False)
    print(f"\nResults saved to {args.dataset}_comparison_results.csv")

if __name__ == '__main__':
    args = parse_args()
    compare_models(args)