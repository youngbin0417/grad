import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# CelebA의 사용자 정의 데이터셋 클래스
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, attr_path, partition_path, partition_type, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 이미지 파티션 읽기 (train, val, test)
        self.partition_df = pd.read_csv(partition_path, delim_whitespace=True, header=None)
        self.partition_df.columns = ['image_id', 'partition']
        # 0: train, 1: val, 2: test
        partition_map = {'train': 0, 'val': 1, 'test': 2}
        self.partition_df = self.partition_df[self.partition_df['partition'] == partition_map[partition_type]]
        
        # 속성 읽기
        with open(attr_path, 'r') as f:
            self.attr_info = f.readlines()
        
        # 속성명 가져오기
        self.attr_names = self.attr_info[1].split()
        
        # 이미지 ID와 속성 매핑
        self.attr_df = pd.DataFrame([line.split() for line in self.attr_info[2:]], 
                                    columns=['image_id'] + self.attr_names)
        
        # 데이터 타입 변환
        for attr in self.attr_names:
            self.attr_df[attr] = self.attr_df[attr].astype(int)
        
        # 파티션과 속성 병합
        self.df = pd.merge(self.partition_df, self.attr_df, on='image_id')
        
        # 타겟(Blonde_Hair)과 편향(Male) 속성 인덱스 찾기
        self.target_idx = self.attr_names.index('Blond_Hair')
        self.bias_idx = self.attr_names.index('Male')
        
        print(f"로드된 {partition_type} 이미지 수: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        
        # 이미지 변환
        if self.transform:
            image = self.transform(image)
        
        # 타겟과 편향 속성 가져오기 (-1을 1로, 1을 0으로 변환)
        target = 1 if int(self.df.iloc[idx][self.attr_names[self.target_idx]]) == 1 else 0
        bias = 1 if int(self.df.iloc[idx][self.attr_names[self.bias_idx]]) == 1 else 0
        
        return image, target, bias, img_name

def extract_celeba_features(data_path, output_path, batch_size=32):
    celeba_dir = os.path.join(data_path, "celeba")
    img_dir = os.path.join(celeba_dir, "img_align_celeba")
    attr_path = os.path.join(celeba_dir, "list_attr_celeba.txt")
    partition_path = os.path.join(celeba_dir, "list_eval_partition.txt")
    
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
        dataset = CelebADataset(img_dir, attr_path, partition_path, split, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,pin_memory=True)
        
        features_list = []
        targets_list = []
        biases_list = []
        
        # 특성 추출
        with torch.no_grad():
            for batch_images, batch_targets, batch_biases, _ in tqdm(dataloader):
                batch_images = batch_images.to(device)
                batch_features = feature_extractor(batch_images).squeeze()
                features_list.append(batch_features.detach().cpu().numpy().copy())
                del batch_images, batch_features  # 메모리 즉시 해제
                torch.cuda.synchronize()  # GPU 연산 동기화
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
    output_path = "./datasets/celeba_features"
    extract_celeba_features(data_path, output_path)
