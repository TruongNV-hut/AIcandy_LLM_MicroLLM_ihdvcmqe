import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import argparse
# Check and select device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Create a directory to save the model
os.makedirs('checkpoints', exist_ok=True)

# Global variables
word_to_id = {}
id_to_word = {}
vocab = []
vocab_size = 0

def getVocabFromSentences(sentences):
    """Create a dictionary from the input sentences"""
    vocab_set = set()
    for sentence in sentences:
        words = sentence.split()
        vocab_set.update(word.lower() for word in words)
    vocab_list = sorted(list(vocab_set))
    return vocab_list

def tokenize(sentence, word_to_id):
    """Convert a sentence into a sequence of token IDs"""
    return [word_to_id[word] for word in sentence.lower().split() if word in word_to_id]

def detokenize(tokens, id_to_word):
    """Convert a sequence of token IDs back into a sentence"""
    return ' '.join([id_to_word[token] for token in tokens])

class PositionalEncoding(nn.Module):
    """Positional encoding class for the transformer"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class MicroLLM(nn.Module):
    """MicroLLM Model"""
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(MicroLLM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout)
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc_hidden = nn.Linear(embedding_dim, embedding_dim * 2)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(embedding_dim * 2, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a mask for the decoder"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    def forward(self, src, tgt):
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
        src_mask = None
        
        src = self.dropout(self.embedding(src))
        tgt = self.dropout(self.embedding(tgt))
        
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        src = self.norm1(src)
        tgt = self.norm2(tgt)
        
        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        
        output = self.fc_hidden(output)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.fc_out(output)
        
        return output

def getSentencesFromFile(filePath):
    """Read sentences from a file"""
    with open(filePath, "r", encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences

def generate_data(filePath):
    """Generate training data"""
    global word_to_id, id_to_word, vocab, vocab_size

    sentences = getSentencesFromFile(filePath)
    
    vocab = getVocabFromSentences(sentences)
    vocab_size = len(vocab)
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for idx, word in enumerate(vocab)}

    data = []
    max_len = max(len(tokenize(sentence, word_to_id)) for sentence in sentences)
    
    for sentence in sentences:
        tokens = tokenize(sentence, word_to_id)
        
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        
        input_seq = input_seq + [0] * (max_len - len(input_seq))
        target_seq = target_seq + [0] * (max_len - len(target_seq))
        
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        target_tensor = torch.tensor(target_seq, dtype=torch.long).unsqueeze(0).to(device)
        
        data.append((input_tensor, target_tensor))
    
    return data

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save the model state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'vocab': vocab,
        'vocab_size': vocab_size
    }
    torch.save(checkpoint, filename)
    print(f"Epoch: {epoch}, saved checkpoint at {filename}, loss: {loss}")

def load_checkpoint(filename, model, optimizer=None):
    """Load the model state"""
    global word_to_id, id_to_word, vocab, vocab_size
    
    if not os.path.exists(filename):
        print(f"Checkpoint {filename} does not exist!")
        return 0, float('inf')
    
    try:
        # Add weights_only=True for enhanced security
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
        
        # Check that required keys exist in the checkpoint
        required_keys = ['word_to_id', 'id_to_word', 'vocab', 'vocab_size', 
                        'model_state_dict', 'optimizer_state_dict', 'epoch', 'loss']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Key '{key}' not found in the checkpoint")
        
        word_to_id = checkpoint['word_to_id']
        id_to_word = checkpoint['id_to_word']
        vocab = checkpoint['vocab']
        vocab_size = checkpoint['vocab_size']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {filename}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Loss: {checkpoint['loss']}")
        
        return checkpoint['epoch'], checkpoint['loss']
    
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, float('inf')

def train_model(model, training_data, criterion, optimizer, num_epochs, pathCheckpoint="checkpoint.pt"):
    """Train the model"""
    best_loss = float('inf')
    start_epoch = 0
    
    # Update scheduler initialization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    if os.path.exists(pathCheckpoint):
        start_epoch, best_loss = load_checkpoint(pathCheckpoint, model, optimizer)
    
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        model.train()
        
        for src, tgt in training_data:
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(training_data) if len(training_data) > 0 else float('inf')
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}')
        
        # Update scheduler
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss and not (torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss))):
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch + 1, best_loss, pathCheckpoint)
    
    return best_loss


def test_model(checkpoint_path, input_text, max_length=50, temperature=0.7):
    """
    Test sentence generation with a trained model
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        input_text (str): Input sentence to generate the next sequence
        max_length (int): Maximum length of the generated sentence
        temperature (float): Temperature for sentence generation (controls randomness)
    """
    try:
        # Check if the checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # Extract parameters from the checkpoint
        word_to_id = checkpoint['word_to_id']
        id_to_word = checkpoint['id_to_word']
        vocab_size = checkpoint['vocab_size']
        
        # Initialize the model with the same parameters as during training
        model = MicroLLM(
            vocab_size=vocab_size,
            embedding_dim=256,
            num_heads=16,
            num_layers=12,
            dropout=0.1
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate sentence with additional parameters
        def generate_sentence(model, input_text, word_to_id, id_to_word, max_length=max_length, temperature=temperature):
            model.eval()
            with torch.no_grad():
                input_tokens = [word_to_id.get(word.lower(), 0) for word in input_text.split()]
                input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
                
                output_tokens = input_tokens.copy()
                
                for _ in range(max_length):
                    tgt_tensor = torch.tensor(output_tokens, dtype=torch.long).unsqueeze(0).to(device)
                    
                    output = model(input_tensor, tgt_tensor)
                    next_token_logits = output[0, -1, :]
                    
                    probs = torch.softmax(next_token_logits / temperature, dim=0)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    output_tokens.append(next_token)
                    
                    # Stop when encountering the end-of-sentence token
                    if next_token == word_to_id.get('.', -1) or len(output_tokens) > max_length:
                        break
                
                generated_words = [id_to_word.get(token, '<unk>') for token in output_tokens]
                return ' '.join(generated_words)
        
        # Generate sentence
        generated_sentence = generate_sentence(model, input_text, word_to_id, id_to_word)
        
        print(f"\nInput: '{input_text}'")
        print(f"Generated sentence: '{generated_sentence}'")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='AIcandy.vn - LLM Training and Testing Script')
    parser.add_argument('--mode', 
                        type=str, 
                        choices=['train', 'test'], 
                        required=True, 
                        help='Operation mode: train or test')
    
    parser.add_argument('--input', 
                        type=str, 
                        default=None, 
                        help='Input sentence to test (only applicable when in test mode))')
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Hyperparameters
        embedding_dim = 256
        num_heads = 16
        num_layers = 12
        learning_rate = 0.0003
        num_epochs = 15
        dropout_rate = 0.1
        fileDataPath = "datasets/aicandy_llm_dataset_qmvqmhro.txt"
        
        # Generate training data
        training_data = generate_data(fileDataPath)
        
        # Initialize model
        model = MicroLLM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout_rate
        ).to(device)
        
        # Initialize loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Train model
        pathCheckpoint = os.path.join('checkpoints', "checkpoint.pt")
        train_model(model, training_data, criterion, optimizer, num_epochs, pathCheckpoint)
        print("Model training completed!")
    
    elif args.mode == 'test':
        # Path to checkpoint
        checkpoint_path = os.path.join('checkpoints', "checkpoint.pt")
        test_sentences = []
        
        if args.input:
            test_sentences = [args.input]
        
        # Test with each sentence
        print("Starting sentence generation testing:")
        print("-" * 50)
        
        for test_sentence in test_sentences:
            test_model(checkpoint_path, test_sentence, max_length=10)
            print("-" * 50)
    
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")


if __name__ == "__main__":
    main()