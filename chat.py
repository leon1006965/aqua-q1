import torch
import json
from model import MiniGPT

def load_model():
    with open("vocab.json", 'r') as f:
        stoi = json.load(f)
    itos = {i: ch for ch, i in stoi.items()}

    checkpoint = torch.load("brain.pt", map_location='cpu', weights_only=False)
    model = MiniGPT(
        checkpoint['vocab_size'],
        checkpoint['embed_dim'],
        checkpoint['num_heads'],
        checkpoint['num_layers'],
        checkpoint['block_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, stoi, itos, checkpoint['block_size']

def generate_greedy(model, stoi, itos, prompt, max_new=400, block_size=128):
    encoded = [stoi[c] for c in prompt.lower() if c in stoi]
    if not encoded:
        return ""
    context = torch.tensor([encoded[-block_size:]], dtype=torch.long)
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(context)[0, -1, :]
            next_token = torch.argmax(logits).item()
            chosen = itos[next_token]
            generated.append(chosen)
            if chosen == '\n':
                break
            context = torch.cat([context[:, 1:], torch.tensor([[next_token]])], dim=1)
    return ''.join(generated)

def chat():
    model, stoi, itos, block_size = load_model()
    print("=" * 50)
    print("  Aqua AI - Made by Kael Computing")
    print("  Model: Aqua Q1")
    print("  Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAqua AI: Goodbye!")
            break
        if not user:
            continue
        if user.lower() in ['quit', 'exit', 'q']:
            print("Aqua AI: Goodbye!")
            break

        response = generate_greedy(model, stoi, itos, user + " ->", block_size=block_size).strip()
        if not response or len(response) < 3:
            response = generate_greedy(model, stoi, itos, "hello ->", block_size=block_size).strip()
        if not response or len(response) < 3:
            response = "I am Aqua Q1 by Kael Computing. I can build websites in HTML."

        print(f"Aqua AI: {response}\n")

if __name__ == "__main__":
    chat()