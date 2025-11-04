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
        for _, (group_id, features, targets, z1, _) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            
            if model_type == 'baseline':
                logits, _, _ = model(features)
            elif model_type == 'margin':
                logits, _, _, _, _ = model(features, margin_model, s=config.scale)
            elif model_type == 'ensemble':
                # For ensemble models, may need special handling
                logits, _, _, _, _ = model(features, margin_model, s=config.scale)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_group_ids.extend(group_id.numpy())
    
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
        for _, (group_id, features, targets, z1, _) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            
            # Collect logits from all models in the ensemble
            batch_logits_list = []
            for model in models:
                if model_type == 'baseline':
                    logits, _, _ = model(features)
                else:  # margin
                    logits, _, _, _, _ = model(features, None, s=config.scale)  # margin parameter may vary
                batch_logits_list.append(logits.unsqueeze(0))
            
            # Average logits across ensemble models
            avg_logits = torch.cat(batch_logits_list, dim=0).mean(dim=0)
            
            all_logits.append(avg_logits.cpu())
            all_targets.extend(targets.cpu().numpy())
            all_group_ids.extend(group_id.numpy())
    
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
        'num_class': config.num_class,
        'mlp_neurons': config.mlp_neurons,
        'hid_dim': config.hid_dim
    }
    
    margin_kwargs = {
        'model_name': config.model_name,
        'num_class': config.num_class,
        'device': device,
        'std': config.std,
        'mlp_neurons': config.mlp_neurons,
        'hid_dim': config.hid_dim
    }
    
    results = []
    
    # Load and evaluate baseline model
    if os.path.exists(config.basemodel_path):
        baseline_model = load_model(Network, config.basemodel_path, device, **baseline_kwargs)
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
    if os.path.exists(config.margin_path):
        margin_model = load_model(NetworkMargin, config.margin_path, device, **margin_kwargs)
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
    
    # Try to load and evaluate ensemble models
    # For EMA ensemble
    ema_ensemble_paths = [f'ensemble_model/ema_baseline_epoch{i}.pt' for i in range(config.base_epochs)]
    ema_ensemble_paths = [p for p in ema_ensemble_paths if os.path.exists(p)]
    if ema_ensemble_paths:
        ema_ensemble_models = []
        for path in ema_ensemble_paths[:5]:  # Use top 5 models (or fewer if not available)
            try:
                model = load_model(Network, path, device, **baseline_kwargs)
                ema_ensemble_models.append(model)
            except:
                continue
        if ema_ensemble_models:
            global_acc, worst_acc, avg_acc, group_accs = evaluate_ensemble(
                ema_ensemble_models, test_loader, device, model_type='baseline')
            results.append(['Baseline-EMA-Ensemble', global_acc, worst_acc, avg_acc])
    
    # For SWA ensemble
    swa_ensemble_paths = [f'ensemble_model/swa_baseline_epoch{i}.pt' for i in range(config.base_epochs)]
    swa_ensemble_paths = [p for p in swa_ensemble_paths if os.path.exists(p)]
    if swa_ensemble_paths:
        swa_ensemble_models = []
        for path in swa_ensemble_paths[:5]:  # Use top 5 models (or fewer if not available)
            try:
                model = load_model(Network, path, device, **baseline_kwargs)
                swa_ensemble_models.append(model)
            except:
                continue
        if swa_ensemble_models:
            global_acc, worst_acc, avg_acc, group_accs = evaluate_ensemble(
                swa_ensemble_models, test_loader, device, model_type='baseline')
            results.append(['Baseline-SWA-Ensemble', global_acc, worst_acc, avg_acc])
    
    # For margin + EMA ensemble
    ema_margin_ensemble_paths = [f'ensemble_model/ema_margin_epoch{i}.pt' for i in range(config.base_epochs)]
    ema_margin_ensemble_paths = [p for p in ema_margin_ensemble_paths if os.path.exists(p)]
    if ema_margin_ensemble_paths:
        ema_margin_ensemble_models = []
        for path in ema_margin_ensemble_paths[:5]:  # Use top 5 models (or fewer if not available)
            try:
                model = load_model(NetworkMargin, path, device, **margin_kwargs)
                ema_margin_ensemble_models.append(model)
            except:
                continue
        if ema_margin_ensemble_models:
            global_acc, worst_acc, avg_acc, group_accs = evaluate_ensemble(
                ema_margin_ensemble_models, test_loader, device, model_type='margin')
            results.append(['Margin-EMA-Ensemble', global_acc, worst_acc, avg_acc])
    
    # For margin + SWA ensemble
    swa_margin_ensemble_paths = [f'ensemble_model/swa_margin_epoch{i}.pt' for i in range(config.base_epochs)]
    swa_margin_ensemble_paths = [p for p in swa_margin_ensemble_paths if os.path.exists(p)]
    if swa_margin_ensemble_paths:
        swa_margin_ensemble_models = []
        for path in swa_margin_ensemble_paths[:5]:  # Use top 5 models (or fewer if not available)
            try:
                model = load_model(NetworkMargin, path, device, **margin_kwargs)
                swa_margin_ensemble_models.append(model)
            except:
                continue
        if swa_margin_ensemble_models:
            global_acc, worst_acc, avg_acc, group_accs = evaluate_ensemble(
                swa_margin_ensemble_models, test_loader, device, model_type='margin')
            results.append(['Margin-SWA-Ensemble', global_acc, worst_acc, avg_acc])
    
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