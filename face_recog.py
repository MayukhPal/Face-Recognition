import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm, trange

class RecursiveFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        person_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        person_folders.sort()
        self.k4_labels = [f"K{idx+1:04d}" for idx in range(len(person_folders))]
        self.folder_to_k4 = {folder: k4 for folder, k4 in zip(person_folders, self.k4_labels)}
        self.k4_to_idx = {k4: idx for idx, k4 in enumerate(self.k4_labels)}
        for person in person_folders:
            person_dir = os.path.join(root_dir, person)
            k4_label = self.folder_to_k4[person]
            label_idx = self.k4_to_idx[k4_label]
            for subdir, _, files in os.walk(person_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(subdir, file)
                        self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping corrupted image: {img_path} | Error: {e}")
            image = Image.new("RGB", (112, 112))
        if self.transform:
            image = self.transform(image)
        return image, label

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in backbone.parameters():
            param.requires_grad = True
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for images, labels in tqdm(dataloader, desc='🟢 Training', leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='🔵 Validation', leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1

def predict(model, dataloader, device, idx_to_label=None):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    if idx_to_label:
        preds = [idx_to_label[p] for p in preds]
    return preds

def main(train_dir, val_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = RecursiveFaceDataset(train_dir, train_transform)
    val_dataset = RecursiveFaceDataset(val_dir, val_transform)

    print(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(train_dataset.k4_to_idx)
    model = FaceRecognitionModel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0
    for epoch in trange(10, desc="Epochs"):
        print(f"\nEpoch {epoch + 1}/10")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} F1: {train_f1:.4f} | Val Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_face_model_real.pth")

    print(f"\n✅ Training done. Best Val Accuracy: {best_acc:.4f}")
    idx_to_k4 = {v: k for k, v in train_dataset.k4_to_idx.items()}
    return model, val_loader, idx_to_k4

def inference_example(model, val_loader, idx_to_k4, device):
    preds = predict(model, val_loader, device, idx_to_k4)
    print("\n🧪 Sample predictions:", preds[:10])

if __name__ == "__main__":
    train_dir = r"C:\Users\91974\Downloads\Comys_Hackathon5\Comys_Hackathon5\Task_B\train"
    val_dir = r"C:\Users\91974\Downloads\Comys_Hackathon5\Comys_Hackathon5\Task_B\val"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, val_loader, idx_to_k4 = main(train_dir, val_dir)
    inference_example(model, val_loader, idx_to_k4, device)