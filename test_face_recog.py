import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

class RecursiveFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, k4_to_idx=None):
        self.transform = transform
        self.samples = []
        self.k4_to_idx = k4_to_idx or {}
        person_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if not self.k4_to_idx:
            k4_labels = [f"K{idx+1:04d}" for idx in range(len(person_folders))]
            self.k4_to_idx = {k4: idx for idx, k4 in enumerate(k4_labels)}
            self.folder_to_k4 = {folder: k4 for folder, k4 in zip(person_folders, k4_labels)}
        else:
            self.folder_to_k4 = {folder: k4 for folder, k4 in zip(person_folders, self.k4_to_idx.keys())}

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
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="🔍 Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')

    print(f"\n✅ Test Accuracy: {acc:.4f} | Macro F1-Score: {f1:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(labels, preds))
    return acc, f1

if __name__ == "__main__":
    test_dir = r"C:\Users\91974\Downloads\Comys_Hackathon5\Comys_Hackathon5\Task_B\val"
    model_path = "best_face_model_real.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load label mapping from training folder
    temp_train_dir = r"C:\Users\91974\Downloads\Comys_Hackathon5\Comys_Hackathon5\Task_B\train"
    temp_dataset = RecursiveFaceDataset(temp_train_dir)
    k4_to_idx = temp_dataset.k4_to_idx

    test_dataset = RecursiveFaceDataset(test_dir, transform=transform, k4_to_idx=k4_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = FaceRecognitionModel(num_classes=len(k4_to_idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    evaluate(model, test_loader, device)
