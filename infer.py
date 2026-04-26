import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import argparse
import sys


# Define the TC-PED model (same as in notebook)
class TCPEDModel(nn.Module):
    def __init__(self, base_model, hidden_dim=768, hidden_units=128, dropout=0.2):
        super(TCPEDModel, self).__init__()
        self.base_model = base_model
        self.cls_linear = nn.Linear(hidden_dim, hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_units, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Use CLS token embedding
        cls_emb = last_hidden_state[:, 0, :]  # (batch, hidden_dim)

        x = self.cls_linear(cls_emb)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch, 1)
        return logits.squeeze(-1)  # (batch,)


def load_model(model_path, device):
    """Load the trained TC-PED model"""
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
    base_model = AutoModel.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
    
    model = TCPEDModel(base_model, hidden_units=128, dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer


def predict(model, tokenizer, tweet, party, device, max_length=128):
    """Predict if party is mentioned in tweet"""
    encoding = tokenizer(
        tweet,
        party,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
    
    return pred.item(), prob.item()


def main():
    parser = argparse.ArgumentParser(description='TC-PED Model Inference CLI')
    parser.add_argument('model_path', type=str, help='Path to the trained model file (.pth)')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, device)
    print("Model loaded successfully!\n")
    
    # Interactive inference loop
    print("=" * 60)
    print("TC-PED Model - Party Mention Detection")
    print("=" * 60)
    print("Enter 'quit' to exit\n")
    
    while True:
        try:
            # Get tweet input
            tweet = input("Enter tweet text: ").strip()
            if tweet.lower() == 'quit':
                print("Exiting...")
                break
            
            if not tweet:
                print("Tweet cannot be empty. Please try again.\n")
                continue
            
            # Get party input
            party = input("Enter party name: ").strip()
            if party.lower() == 'quit':
                print("Exiting...")
                break
            
            if not party:
                print("Party name cannot be empty. Please try again.\n")
                continue
            
            # Make prediction
            pred, prob = predict(model, tokenizer, tweet, party, device)
            
            # Display result
            result = "MENTIONED" if pred == 1 else "NOT MENTIONED"
            confidence = prob if pred == 1 else (1 - prob)
            
            print(f"\nResult: {result}")
            print(f"Confidence: {confidence:.4f}\n")
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
