# ----------------------Common Hyperparams-------------------------- #
num_class = 1
mlp_neurons = 128

# ----------------------Baseline Hyperparams-------------------------- #
base_epochs = 100
base_batch_size = 64
base_lr = 0.01
weight_decay = 0.01 # Vary this to train a bias-amplified model'
scale = 4
std = 0.2
K = 4

opt_b = 'adam'
opt_m = 'adam'

hid_dim = 512
# ----------------------Paths-------------------------- #
# Paths for model checkpoints.
# The naming convention is: {model_type}_{technique}.pth
# model_type: 'baseline' or 'margin'
# technique: 'erm' (standard), 'ema', or 'swa'

# Standard ERM training
baseline_path_erm = './erm/baseline_erm.pth'
margin_path_erm = './erm/margin_erm.pth'

# EMA training
baseline_path_ema = './ema/baseline_ema.pth'
margin_path_ema = './ema/rgin_ema.pth'

# SWA training
baseline_path_swa = './swa/baseline_swa.pth'
margin_path_swa = './swa/margin_swa.pth'

# Path for the pre-trained baseline model used by all CAML variants
# This MUST point to the standard ERM baseline for correct margin calculation
basemodel_path_for_margin = './erm/baseline_erm.pth'
basemodel_path = './erm/baseline_erm.pth'




# ----------------------Model-details-------------------------- #
model_name = 'resnet18'

# ----------------------ImageNet Means and Transforms---------- #
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]

# -----------------------CelebA/Waterbirds-parameters--------- #
dataset_path = './datasets'

"""
img_dir = './datasets/CelebA/img/img_align_celeba'
partition_path = './datasets/CelebA/Eval/list_eval_partition.txt'
attr_path = './datasets/CelebA/Anno/list_attr_celeba.txt'
"""
img_dir = './datasets/waterbirds/waterbird_complete95_forest2water2'
metadata_path = './datasets/waterbirds/waterbird_complete95_forest2water2/metadata.csv'


#target_attribute = 'Blond_Hair'
#bias_attribute = 'Male'

target_attribute = 'y'         # Bird type: 0 = landbird, 1 = waterbird
bias_attribute = 'place'      # Background: 0 = land, 1 = water


celeba_path = './dataset/celeba_features'
celeba_val_path = './dataset/celeba_features'
waterbirds_path = './dataset/waterbirds_features'
waterbirds_val_path = './dataset/waterbirds_features'

# -----------------------Stability options--------- #
use_ema = True
ema_decay = 0.999
grad_clip = 1.0
