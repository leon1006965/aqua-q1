import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import json
import os
import time
from tqdm import tqdm
from model import MiniGPT

DATA_FILE = "data.txt"
MODEL_FILE = "brain.pt"
CHAR_FILE = "vocab.json"

mp.set_start_method('fork', force=True)

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]

def decode(idx, itos):
    return ''.join([itos[i] for i in idx])

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train():
    with open(DATA_FILE, 'r') as f:
        text = f.read()

    print(f"Training on {len(text)} characters...")

    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    with open(CHAR_FILE, 'w') as f:
        json.dump(stoi, f)

    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    block_size = 128
    batch_size = 64
    embed_dim = 384
    num_heads = 8
    num_layers = 10
    lr = 3e-4
    epochs = 8000
    save_every = 200
    num_cores = os.cpu_count()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using device: {device} ({num_cores} cores)")
    print(f"Batch size: {batch_size}")

    model = MiniGPT(vocab_size, embed_dim, num_heads, num_layers, block_size).to(device)
    print("Running in standard mode")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Target: {epochs} epochs (saves every {save_every})")

    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(MODEL_FILE):
        try:
            cp = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
            if 'epoch' in cp:
                model.load_state_dict(cp['model_state_dict'])
                optimizer.load_state_dict(cp['optimizer_state_dict'])
                scheduler.load_state_dict(cp['scheduler_state_dict'])
                start_epoch = cp['epoch'] + 1
                best_loss = cp.get('best_loss', float('inf'))
                print(f"[RESUME] Found checkpoint - continuing from epoch {start_epoch}/{epochs}")
                print(f"         Previous best loss: {best_loss:.4f}")
            elif 'model_state_dict' in cp:
                model.load_state_dict(cp['model_state_dict'])
                print("[RESUME] Found old-format checkpoint (weights only)")
                print("         Loaded trained weights, continuing training")
                print("         (Epoch counter resets but ALL learned weights are kept)")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")
            print("         Starting fresh")

    print(f"Starting training for {epochs} epochs...\n")

    start = time.time()
    pbar = tqdm(range(start_epoch, epochs), desc="Training", ncols=100, initial=start_epoch, total=epochs)
    for epoch in pbar:
        xb, yb = get_batch(data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

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

        if epoch % 500 == 0 and epoch > 0:
            elapsed = time.time() - start
            speed = (epoch - start_epoch + 1) / elapsed
            remaining = (epochs - epoch - 1) / speed
            print(f"\n  [Epoch {epoch:5d}] loss={current_loss:.4f} | best={best_loss:.4f} | speed={speed:.2f}it/s | ETA={remaining/60:.1f}min")

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'vocab_size': vocab_size,
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
