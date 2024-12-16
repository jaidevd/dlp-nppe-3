import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import EnhanceDataset, collate_denoise
from mprnet import MPRNet, CharbonnierLoss
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, optimizer and loss
model = MPRNet().to(device)
optimizer = optim.Adam(
    model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=1e-8
)
criterion = CharbonnierLoss()

# datasets and loaders
batch_size = 3
train_dataset = EnhanceDataset("archive/train/train", "archive/train/gt")
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_denoise, pin_memory=True
)

val_dataset = EnhanceDataset("archive/val/val", "archive/val/gt")
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=True,
    collate_fn=collate_denoise, pin_memory=True
)

# Training parameters
n_epochs = 30
stepper = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


for epoch in range(n_epochs):
    epoch_train_loss = epoch_val_loss = 0
    with tqdm(total=len(train_loader) + len(val_loader)) as pbar:
        model.train()
        for input_, target in train_loader:
            optimizer.zero_grad()
            restored = model(input_.to(device))
            loss = sum(
                [
                    criterion(torch.clamp(restored[j], 0, 1), target.to(device))
                    for j in range(len(restored))
                ]
            )
            loss.backward()
            optimizer.step()
            litem = loss.item()
            epoch_train_loss += litem
            pbar.set_postfix_str(f"train loss: {litem:.4f}")
            pbar.update(1)

        model.eval()
        for input_, target in val_loader:
            with torch.no_grad():
                restored = model(input_.to(device))
            loss = sum(
                [
                    criterion(torch.clamp(restored[j], 0, 1), target.to(device))
                    for j in range(len(restored))
                ]
            )
            litem = loss.item()
            epoch_val_loss += litem
            pbar.set_postfix_str(f"val loss: {litem:.4f}")
            pbar.update(1)
        stepper.step()

    print(  # NOQA: T201
        "Epoch {} Train Loss: {:.4f}; Val Loss: {:.4f}".format(
            epoch,
            epoch_train_loss / len(train_loader),
            epoch_val_loss / len(val_loader),
        )
    )
os.makedirs("models", exist_ok=True)
torch.save(model, "models/mprnet-denoiser.pt")
