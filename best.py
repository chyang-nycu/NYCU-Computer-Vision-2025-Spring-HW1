import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datetime import datetime
import zipfile
import numpy as np
import random

NUM_EPOCHS = 30
BATCH_SIZE = 80
LR = 2e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CUTMIX_PROB = 0.3  # CutMix的概率

start_time = datetime.now()
current_time = start_time.strftime("%d_%H_%M")
properties = f"calr_CutMix{CUTMIX_PROB}_e{NUM_EPOCHS}"
output_file = f"{current_time}_{properties}"

output_dir = "./kaggle/working/"
os.makedirs(output_dir, exist_ok=True)  # 確保目錄存在

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"當前使用裝置: {device}")
# num_gpus = torch.cuda.device_count()
num_gpus = 1
print(f"Using {num_gpus} GPUs")

###########################################
# 1. 透過 ImageFolder 建立資料集 (train/val)
###########################################
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(root='data/train', transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(root='data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

NUM_CLASSES = len(train_dataset.classes)
print("Number of classes:", NUM_CLASSES)

###################################
# 2. 載入 ResNeXt-101 64×4d 模型
###################################
from torchvision.models import ResNeXt101_64X4D_Weights

model = torchvision.models.resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

# **關鍵: 使用 DataParallel**
if num_gpus > 1:
    model = nn.DataParallel(model)

model = model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model Parameters: {total_params/1e6:.2f}M")

###################################################
# 3. 設定損失函式、優化器、學習率排程 (Scheduler)
###################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


########################################################
# 4. 訓練 (Train) + 驗證 (Val) 迴圈
########################################################

# 添加 CutMix 相关函数
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 边界框
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

best_val_acc = 0.0
best_model_weights = None

print("Training...")
for epoch in range(NUM_EPOCHS):
    # ---------------------
    # Training
    # ---------------------
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 应用 CutMix
        r = np.random.rand(1)
        if r < CUTMIX_PROB:
            # 生成混合参数
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            
            # 获取目标标签
            target_a = labels
            target_b = labels[rand_index]
            
            # 生成混合区域
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            
            # 执行混合
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            
            # 调整混合比例
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算混合损失
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            # 正常前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    # ---------------------
    # Validation
    # ---------------------
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            _, val_preds = torch.max(val_outputs, 1)

            val_running_loss += val_loss.item() * val_inputs.size(0)
            val_running_corrects += torch.sum(val_preds == val_labels.data)

    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_running_corrects.double() / len(val_dataset)

    scheduler.step()

    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        best_model_weights = model.state_dict()

    print(f"[Epoch {epoch+1:02}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.5f} Acc: {epoch_acc:.5f} | "
          f"Val Loss: {val_epoch_loss:.5f} Acc: {val_epoch_acc:.5f}")

# 訓練完成後，載入最佳權重
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    print(f"Loaded best model weights with Val Acc: {best_val_acc:.4f}")

training = datetime.now()
print(f"Training time: {training - start_time}, {(training - start_time)/NUM_EPOCHS} per epoch")

###################################
# 6. 創建測試數據集和加載器
###################################
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# 測試集的轉換與驗證集相同
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 創建測試數據集和加載器
test_dataset = TestDataset(test_dir='data/test', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

###################################
# 7. 推論 (Inference) 在測試集上
###################################
model.eval()
predictions = []
classnames = []

with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        for i, img_name in enumerate(img_names):
            img_name = os.path.splitext(img_name)[0]  # 確保 `img_name` 是字串
            pred_class_id = preds[i].item()
            predictions.append((img_name, train_dataset.classes[pred_class_id]))

prediction_file = os.path.join(output_dir, "prediction.csv")
zip_file = os.path.join(output_dir, f"{output_file}.zip")

# 8. 輸出競賽需要的檔案 
with open(prediction_file, "w") as f:
    f.write("image_name,pred_label\n")
    for img_name, pred_id in predictions:
        f.write(f"{img_name},{pred_id}\n")

# 把 csv 壓成 zip
with zipfile.ZipFile(zip_file, "w") as zipf:
    zipf.write(prediction_file, arcname="prediction.csv")

print(f"所有預測結果已儲存至 {zip_file}")

end_time = datetime.now()
print(f"Total time: {end_time - start_time}, {(end_time - start_time)/NUM_EPOCHS} per epoch")
