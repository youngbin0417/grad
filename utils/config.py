# ----------------------Common Hyperparams-------------------------- #
num_class = 1
mlp_neurons = 128

# ----------------------Baseline Hyperparams-------------------------- #
base_epochs = 15
base_batch_size = 512
base_lr = 0.0001
weight_decay = 0.1 # Vary this to train a bias-amplified model'
scale = 8
std = 0.15
K = 2

opt_b = 'sgd'
opt_m = 'sgd'

hid_dim = 512
mlp_neurons = 128
# ----------------------Paths-------------------------- #
basemodel_path = 'basemodel.pth' #'{}_{}_base_balanced.pth'.format(bias_attribute, target_attribute)
margin_path = 'margin.pth' #'{}_{}_adv_balanced.pth'.format(bias_attribute, target_attribute)

# ----------------------Model-details-------------------------- #
model_name = 'resnet18'

# ----------------------ImageNet Means and Transforms---------- #
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]

# -----------------------CelebA/Waterbirds-parameters--------- #
dataset_path = './dataset'
img_dir = './dataset/celeba/img_align_celeba'
partition_path = './dataset/celeba/list_eval_partition.txt'
attr_path = './dataset/celeba/list_attr_celeba.txt'
target_attribute = 'Blond_Hair'
bias_attribute = 'Male'
celeba_path = './dataset/celeba_features'
celeba_val_path = './dataset/celeba_features'
waterbirds_path = './dataset/waterbirds_features'
waterbirds_val_path = './dataset/waterbirds_features'
