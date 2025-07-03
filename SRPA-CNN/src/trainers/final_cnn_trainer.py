from torch import nn

from src.models.final_3dcnn import Final3DCNN


def train_final_3dcnn(patches, labels, num_classes=3, epochs=20, batch_size=16, save_path=None):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset, random_split
    from sklearn.metrics import accuracy_score, f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)  # (B, 1, 50, 50, k)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    num_bands = X.shape[-1]
    model = Final3DCNN(num_bands, num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds, targets = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(torch.argmax(model(xb), dim=1).cpu().numpy())
                targets.append(yb.numpy())

            preds = np.concatenate(preds)
            targets = np.concatenate(targets)
            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average="macro")

        print(f"Epoch {epoch + 1}/{epochs} | Val Acc: {acc:.4f}, F1: {f1:.4f}")

    import os

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Now save
    torch.save(model.state_dict(), save_path)

    # if save_path:
    #     torch.save(model.state_dict(), save_path)

    return model, acc, f1
