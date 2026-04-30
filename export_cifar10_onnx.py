import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ---------- MODEL (slightly stronger) ----------
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# ---------- TRAIN ----------
def train_model(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(25):   # 🔥 increase from 15 → 25
        total_loss = 0
        for imgs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")


# ---------- EXPORT ----------
def export_onnx(model):
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model,
        dummy,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )


# ---------- MAIN ----------
if __name__ == "__main__":
    model = BetterCNN()
    train_model(model)
    export_onnx(model)