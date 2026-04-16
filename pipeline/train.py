# Author: proud cheetah group

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Adjust imports for models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.transformer import TextTransformer
from models.autoencoder import SongAutoencoder

class CharTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 -,;'\"()[]{}&|!@#$%^*+=_~`.<>?/\\" 
        self.char2idx = {c: i+1 for i, c in enumerate(chars)} # 0 is padding
        self.char2idx['<UNK>'] = len(self.char2idx) + 1
        self.vocab_size = len(self.char2idx) + 1

    def encode(self, text, max_len=64):
        text = str(text).lower()
        indices = [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text[:max_len]]
        padding = [0] * (max_len - len(indices))
        return indices + padding

class SongDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.audio_features = df[['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'tempo']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['name']} {row['artists']}"
        tokens = self.tokenizer.encode(text, self.max_len)
        features = self.audio_features[idx]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(features, dtype=torch.float32)

def info_nce_loss(text_embeds, audio_embeds, temperature=0.07):
    # Calculate cosine similarity
    logits = torch.matmul(text_embeds, audio_embeds.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_t = F.cross_entropy(logits, labels)
    loss_a = F.cross_entropy(logits.T, labels)
    return (loss_t + loss_a) / 2

def create_mock_data(path, num_rows=10000):
    print("Creating mock dataset for training...")
    np.random.seed(42)
    names = ["Midnight", "Sunlight", "Vibe", "Chill", "Dance", "Groove", "Sadness", "Happy", "Acoustic", "Electric"]
    artists = ["Artist A", "Artist B", "Artist C", "Artist D"]
    
    data = {
        'id': [f'id_{i}' for i in range(num_rows)],
        'name': np.random.choice(names, num_rows),
        'album': [f'Album {i%50}' for i in range(num_rows)],
        'artists': np.random.choice(artists, num_rows),
        'explicit': np.random.randint(0, 2, num_rows),
        'danceability': np.random.rand(num_rows),
        'energy': np.random.rand(num_rows),
        'valence': np.random.rand(num_rows),
        'acousticness': np.random.rand(num_rows),
        'speechiness': np.random.rand(num_rows),
        'tempo': np.random.rand(num_rows) * 200 # Will strictly normalize
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def main():
    data_path = "data/tracks_features_small.csv"
    if not os.path.exists(data_path):
        print(f"Error: dataset not found at {data_path}. Please run pipeline/reduce_dataset.py first.")
        return

    # Memory Management: float32, int8
    # Phase 1: Preprocessing
    dtypes = {
        'danceability': 'float32',
        'energy': 'float32',
        'valence': 'float32',
        'acousticness': 'float32',
        'speechiness': 'float32',
        'tempo': 'float32'
    }
    
    print("Loading data...")
    df = pd.read_csv(data_path, dtype=dtypes)
    if 'explicit' in df.columns:
        df['explicit'] = df['explicit'].astype(bool).astype('int8')
    else:
        df['explicit'] = 0
        
    df = df.dropna(subset=['name', 'artists'])
    
    # Fill remaining NaNs for audio features with 0
    audio_cols = ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'tempo']
    df[audio_cols] = df[audio_cols].fillna(0)
    
    # Standardize tempo to 0.0 - 1.0 (Min-Max Scaling)
    tempo_min = df['tempo'].min()
    tempo_max = df['tempo'].max()
    df['tempo'] = (df['tempo'] - tempo_min) / (tempo_max - tempo_min + 1e-8)
    
    # Save processed metadata as parquet
    parquet_path = "data/processed_songs.parquet"
    print(f"Saving processed data to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    
    tokenizer = CharTokenizer()
    dataset = SongDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    text_encoder = TextTransformer(vocab_size=tokenizer.vocab_size).to(device)
    audio_autoencoder = SongAutoencoder(input_dim=6).to(device)

    optimizer_text = torch.optim.Adam(text_encoder.parameters(), lr=1e-3)
    optimizer_audio = torch.optim.Adam(audio_autoencoder.parameters(), lr=1e-3)

    epochs = 5
    print("Starting Training...")
    for epoch in range(epochs):
        text_encoder.train()
        audio_autoencoder.train()
        total_loss = 0
        total_recon_loss = 0
        
        for batch_idx, (tokens, features) in enumerate(dataloader):
            tokens, features = tokens.to(device), features.to(device)

            optimizer_text.zero_grad()
            optimizer_audio.zero_grad()

            text_embeds = text_encoder(tokens)
            audio_embeds, reconstructions = audio_autoencoder(features)

            # Contrastive Loss (Text <-> Audio)
            c_loss = info_nce_loss(text_embeds, audio_embeds)
            
            # Reconstruction Loss (Audio -> Audio)
            r_loss = F.mse_loss(reconstructions, features)

            # Combined Loss
            loss = c_loss + r_loss
            loss.backward()

            optimizer_text.step()
            optimizer_audio.step()

            total_loss += c_loss.item()
            total_recon_loss += r_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(dataloader)}] "
                      f"Contrastive Loss: {c_loss.item():.4f} recon Loss: {r_loss.item():.4f}")

    print("Training Complete. Saving models...")
    os.makedirs("models/saved", exist_ok=True)
    torch.save(text_encoder.state_dict(), "models/saved/text_model.pth")
    torch.save(audio_autoencoder.state_dict(), "models/saved/audio_model.pth")

if __name__ == "__main__":
    main()
