"""Quick GPU benchmark for the GB10."""
import time
import torch
from transformers import AutoTokenizer, AutoModel

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}")
    else:
        print("WARNING: Running on CPU!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Single example
    inp = tok("This is a test sentence for benchmarking.", return_tensors="pt",
              padding="max_length", max_length=128)
    inp = {k: v.to(device) for k, v in inp.items()}

    # Warmup
    for _ in range(3):
        out = model(**inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Forward only
    t0 = time.time()
    for _ in range(100):
        out = model(**inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 100
    print(f"BERT forward (1 example): {dt*1000:.1f}ms")

    # Forward + backward
    t0 = time.time()
    for _ in range(50):
        model.zero_grad()
        out = model(**inp)
        loss = out.last_hidden_state.mean()
        loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 50
    print(f"BERT fwd+bwd (1 example): {dt*1000:.1f}ms")

    # Batch of 16
    batch = tok(["This is test sentence number {}.".format(i) for i in range(16)],
                return_tensors="pt", padding="max_length", max_length=128)
    batch = {k: v.to(device) for k, v in batch.items()}

    t0 = time.time()
    for _ in range(20):
        model.zero_grad()
        out = model(**batch)
        loss = out.last_hidden_state.mean()
        loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    print(f"BERT fwd+bwd (batch=16): {dt*1000:.1f}ms")
    print("Done.")


if __name__ == "__main__":
    main()
