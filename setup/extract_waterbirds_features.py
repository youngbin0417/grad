import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

class WaterbirdsDataset(Dataset):
    def __init__(self, data_dir, metadata_path, split, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 메타데이터 로드
        self.metadata = pd.read_csv(metadata_path)
        
        # split에 해당하는 데이터만 필터링
        split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.metadata = self.metadata[self.metadata['split'] == split_dict[split]]
        
        print(f"로드된 {split} 이미지 수: {len(self.metadata)}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # 이미지 경로 가져오기
        img_filename = self.metadata.iloc[idx]['img_filename']
        img_path = os.path.join(self.data_dir, img_filename)
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 이미지 변환
        if self.transform:
            image = self.transform(image)
        
        # 타겟(y)과 편향(a) 가져오기
        target = int(self.metadata.iloc[idx]['y'])  # y: 0 = landbird, 1 = waterbird
        bias = int(self.metadata.iloc[idx]['place'])    # a: 0 = land background, 1 = water background
        
        return image, target, bias, img_filename

def extract_waterbirds_features(data_path, output_path, batch_size=32):
    waterbirds_dir = os.path.join(data_path, "waterbirds")
    waterbirds_dataset_dir = os.path.join(waterbirds_dir, "waterbird_complete95_forest2water2")
    metadata_path = os.path.join(waterbirds_dataset_dir, "metadata.csv")
    
    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 사전 훈련된 모델 로드
    model = models.resnet18(pretrained=True)
    # 마지막 fully connected 레이어 제거 (특성 추출용)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # 각 데이터 분할 처리 (train, val, test)
    for split in ['train', 'val', 'test']:
        print(f"Processing {split} split...")
        
        # 데이터셋 로드
        dataset = WaterbirdsDataset(waterbirds_dataset_dir, metadata_path, split, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        features_list = []
        targets_list = []
        biases_list = []
        
        # 특성 추출
        with torch.no_grad():
            for batch_images, batch_targets, batch_biases, _ in tqdm(dataloader):
                batch_images = batch_images.to(device)
                batch_features = feature_extractor(batch_images).squeeze()
                features_list.append(batch_features.cpu().numpy())
                targets_list.append(batch_targets.numpy())
                biases_list.append(batch_biases.numpy())
        
        # 결과 병합 및 저장
        all_features = np.concatenate(features_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)
        all_biases = np.concatenate(biases_list, axis=0)
        
        # 결과 저장
        np.save(os.path.join(output_path, f"{split}_feats.npy"), all_features)
        np.save(os.path.join(output_path, f"{split}_targets.npy"), all_targets)
        np.save(os.path.join(output_path, f"{split}_bias.npy"), all_biases)
        
        print(f"Saved {split} features of shape {all_features.shape}")

if __name__ == "__main__":
    data_path = "./datasets"
    output_path = "./datasets/waterbirds_features"
    extract_waterbirds_features(data_path, output_path)
