import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

#GPU Optimisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Processing Unit:",device)
torch.manual_seed(42)

#Input Classification
class train_dataset(Dataset):

    def __init__(self, images, labels):

        images = images.astype(np.float32) / 255.0
        images = images.reshape(-1, 1, 28, 28)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.transform = transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        return image, self.labels[idx]


class test_dataset(Dataset):

    def __init__(self, images):

        images = images.astype(np.float32) / 255.0
        images = images.reshape(-1, 1, 28, 28)
        self.images = torch.from_numpy(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


#CNN Model
class CNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),#Layers Analysis
            torch.nn.ReLU(),#Layer Stacking
            torch.nn.MaxPool2d(2),#Noise Reduction

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),#Layers Analysis
            torch.nn.ReLU(),#Layer Stacking
            torch.nn.MaxPool2d(2),#Noise Reduction
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#Loading Input
def load_data():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    y = train_df["label"].values
    X = train_df.drop(columns=["label"]).values
    X_test = test_df.values 

    train_data = train_dataset(X, y)
    test_data = test_dataset(X_test)

    return train_data, test_data

#Training Model
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#Modal Prediction
def predict(model, loader):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def main():
    print("Loading data...")
    train_data, test_data = load_data()

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("\nStarting training...")
    for epoch in range(1, 31):
        train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        print(f"Epoch {epoch:2d}/30")

    print("\nGenerating predictions on test set...")
    predictions = predict(model, test_loader)

    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": predictions
    })
    submission.to_csv("submission.csv", index=False)
    print("Sucessfully saved the labelled dataset!")


if __name__ == "__main__":
    main()