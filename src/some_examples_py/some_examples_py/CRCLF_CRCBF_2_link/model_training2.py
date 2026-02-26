import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. MODEL DEFINITION ---
class StructuredDynamicsNet(nn.Module):
    def __init__(self, n_dof=2):
        super().__init__()
        self.n_dof = n_dof
        self.backbone = nn.Sequential(
            nn.Linear(n_dof, 128), nn.Softplus(), 
            nn.Linear(128, 256), nn.Softplus(),
            nn.Linear(256, 128), nn.Softplus()
        )
        self.mass_head = nn.Linear(128, (n_dof * n_dof + n_dof) // 2)
        self.h_head = nn.Sequential(
            nn.Linear(128 + n_dof, 128), nn.Softplus(), 
            nn.Linear(128, n_dof)
        )

    def get_matrices(self, q, dq):
        phi = self.backbone(q)
        l_params = self.mass_head(phi)
        batch_size = q.shape[0]
        L = torch.zeros((batch_size, self.n_dof, self.n_dof), device=q.device)
        
        # Stability Epsilon (1e-2) to prevent numerical 'nonsense' spikes
        L[:, 0, 0] = torch.exp(l_params[:, 0]) + 1e-2 
        L[:, 1, 0] = l_params[:, 1]
        L[:, 1, 1] = torch.exp(l_params[:, 2]) + 1e-2
        
        M = torch.bmm(L, L.transpose(1, 2))
        H = self.h_head(torch.cat([phi, dq], dim=-1))
        return M, H

    def forward(self, q, dq, ddq):
        M, H = self.get_matrices(q, dq)
        return torch.bmm(M, ddq.unsqueeze(-1)).squeeze(-1) + H

# --- 2. DATASET & UTILS ---
class RobotDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.q = torch.tensor(df[['q1', 'q2']].values, dtype=torch.float32)
        self.dq = torch.tensor(df[['dq1', 'dq2']].values, dtype=torch.float32)
        self.ddq = torch.tensor(df[['ddq1', 'ddq2']].values, dtype=torch.float32)
        self.tau = torch.tensor(df[['tau1', 'tau2']].values, dtype=torch.float32)
    def __len__(self): return len(self.q)
    def __getitem__(self, idx): return self.q[idx], self.dq[idx], self.ddq[idx], self.tau[idx]

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for q, dq, ddq, tau in loader:
            q, dq, ddq, tau = q.to(device), dq.to(device), ddq.to(device), tau.to(device)
            val_loss += criterion(model(q, dq, ddq), tau).item()
    return val_loss / len(loader)

# --- 3. TRAINING LOOP ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "robot_data"
    
    train_set = RobotDataset(f"{DATA_PATH}/train_data.csv")
    val_set = RobotDataset(f"{DATA_PATH}/val_data.csv")
    
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    
    model = StructuredDynamicsNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.HuberLoss()
    
    epochs = 150
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print(f"Starting Training on {DEVICE}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for q, dq, ddq, tau in loop:
            q, dq, ddq, tau = q.to(DEVICE), dq.to(DEVICE), ddq.to(DEVICE), tau.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(q, dq, ddq), tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train = train_loss / len(train_loader)
        avg_val = validate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)

        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.8f} | Val Loss: {avg_val:.8f}")

        # --- SAVE LAST CHECKPOINT ---
        torch.save(model.state_dict(), "2dof_dynamics_model_last.pth")

        # --- SAVE BEST MODEL ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "2dof_dynamics_model_best.pth")
            print(f"  --> New Best Model Saved (Val Loss: {best_val_loss:.8f})")

    # --- PLOT LOSS HISTORY ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Huber)')
    plt.title('Dynamics Model Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    plt.show()

    print(f"Training Complete! Best Val Loss: {best_val_loss:.8f}")