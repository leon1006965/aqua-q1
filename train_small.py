import torch
import torch.nn.functional as F
import json
import os
import time
from model import MiniGPT
from tqdm import tqdm

DATA_FILE = "data_small.txt"
MODEL_FILE = "brain.pt"
CHAR_FILE = "vocab.json"

def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"Vocab size: {len(stoi)}")
    with open(CHAR_FILE, 'w') as f:
        json.dump(stoi, f)
    return stoi

def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)

def get_batch(data, block_size, batch_size, vocab_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train():
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    stoi = build_vocab(text)
    data = encode(text, stoi)
    print(f"Training on {len(text):,} characters")
    print(f"File: {DATA_FILE}")

    block_size = 128
    batch_size = 16
    embed_dim = 192
    num_heads = 4
    num_layers = 6
    lr = 3e-4
    epochs = 2000
    save_every = 100
    num_cores = os.cpu_count()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device} ({num_cores} cores)")
    print(f"Model: embed={embed_dim} heads={num_heads} layers={num_layers}")

    model = MiniGPT(len(stoi), embed_dim, num_heads, num_layers, block_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    torch.set_num_threads(num_cores)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training for {epochs} epochs...\n")

    start = time.time()
    best_loss = float('inf')
    pbar = tqdm(range(epochs), desc="Training", ncols=100)
    for epoch in pbar:
        xb, yb = get_batch(data, block_size, batch_size, len(stoi))
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, len(stoi)), yb.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss

        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'best': f'{best_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })

        if (epoch + 1) % 500 == 0:
            elapsed = time.time() - start
            speed = (epoch + 1) / elapsed
            remaining = (epochs - epoch - 1) / speed
            print(f"\n  [Epoch {epoch + 1:5d}] loss={current_loss:.4f} | best={best_loss:.4f} | speed={speed:.2f}it/s | ETA={remaining/60:.1f}min")

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'vocab_size': len(stoi),
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'block_size': block_size,
            }, MODEL_FILE)
            pbar.write(f"  [Checkpoint saved at epoch {epoch} -> brain.pt]")

    elapsed = time.time() - start
    print(f"\n\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Final loss: {current_loss:.4f} | Best loss: {best_loss:.4f}")
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Vocab saved to {CHAR_FILE}")

if __name__ == "__main__":
    train()