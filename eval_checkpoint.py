"""Quick eval script for a saved checkpoint on dev + test sets."""
import sys
import torch
from transformers import AutoTokenizer
from data.scierc import build_dataloaders
from models.bert_kg_encoder import BertKGExtractor
from eval.triple_f1 import evaluate

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cast2500_seed44_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    _, dev_loader, test_loader = build_dataloaders(tokenizer, batch_size=16, max_length=128)

    model = BertKGExtractor("allenai/scibert_scivocab_uncased").to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["encoder"])
    model.eval()

    step = ckpt.get("step", "?")
    dev_f1 = ckpt.get("metrics", {}).get("triple_f1", 0)
    print(f"Checkpoint: {ckpt_path} (step={step}, saved_dev_f1={dev_f1:.4f})")

    dev_m = evaluate(model, dev_loader, device)
    print(f"DEV:  NER={dev_m['ner_f1']:.4f} RE={dev_m['re_f1']:.4f} Triple={dev_m['triple_f1']:.4f}")

    test_m = evaluate(model, test_loader, device)
    print(f"TEST: NER={test_m['ner_f1']:.4f} RE={test_m['re_f1']:.4f} Triple={test_m['triple_f1']:.4f}")


if __name__ == "__main__":
    main()
