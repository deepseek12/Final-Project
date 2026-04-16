# Author: proud cheetah group
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import streamlit as st
import pandas as pd
import numpy as np
import torch

from models.transformer import TextTransformer

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

# Cache dataset loading
@st.cache_resource
def load_data():
    parquet_path = "data/processed_songs.parquet"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        return df
    else:
        st.warning("Dataset not found! Please run train.py first.")
        # Return empty DF so the app doesn't crash completely
        return pd.DataFrame(columns=['id', 'name', 'artists', 'album', 'danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'tempo'])

# Cache pure Numpy Matrix
@st.cache_resource
def load_vectors():
    index_path = "data/song_vectors.npy"
    if os.path.exists(index_path):
        return np.load(index_path)
    else:
        st.warning("Vectors not found! Please run index_builder.py first.")
        return None

# Cache Text Transformer
@st.cache_resource
def load_transformer():
    tokenizer = CharTokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = TextTransformer(vocab_size=tokenizer.vocab_size).to(device)
    
    model_path = "models/saved/text_model.pth"
    if os.path.exists(model_path):
        encoder.load_state_dict(torch.load(model_path, map_location=device))
    else:
        st.warning("Trained Transformer weights not found! Using random weights.")
    encoder.eval()
    return encoder, tokenizer, device

def main():
    st.set_page_config(page_title="EchoVibe", page_icon="🎵", layout="wide")
    st.title("EchoVibe 🎵")
    st.subheader("Hybrid Semantic-Search Music Engine")
    
    df = load_data()
    vectors = load_vectors()
    encoder, tokenizer, device = load_transformer()
    
    st.sidebar.header("The 6 Pillars of Vibe")
    
    slider_energy = st.sidebar.slider("Vibe Energy", -1.0, 1.0, 0.0, step=0.01, help="Boosts or cuts high-intensity tracks.")
    slider_dance = st.sidebar.slider("Danceability", -1.0, 1.0, 0.0, step=0.01, help="Focuses on rhythm and beat strength.")
    slider_mood = st.sidebar.slider("Mood (Valence)", -1.0, 1.0, 0.0, step=0.01, help="Shifts from Happy/Positive to Sad/Melancholy.")
    slider_acoustic = st.sidebar.slider("Acoustic Feel", -1.0, 1.0, 0.0, step=0.01, help="Favors natural instruments.")
    slider_speech = st.sidebar.slider("Vocal Focus (Speechiness)", -1.0, 1.0, 0.0, step=0.01, help="Shifts between instrumental beats and vocal-heavy tracks.")
    slider_tempo = st.sidebar.slider("Intensity (Tempo)", -1.0, 1.0, 0.0, step=0.01, help="Fine-tunes the speed.")

    query = st.text_input("Describe the vibe you want:", placeholder="e.g. Late night driving...")
    
    if not query:
        st.markdown("### Top Trending Songs")
        if not df.empty:
            # Show top "trending" or random songs if query is empty
            sample_df = df.sample(min(10, len(df)), random_state=42)
            for idx, row in sample_df.iterrows():
                st.markdown(f"**{row['name']}** by *{row['artists']}* (Album: {row['album']})")
                cols = st.columns(5)
                cols[0].metric("Energy", f"{row['energy']:.2f}")
                cols[1].metric("Dance", f"{row['danceability']:.2f}")
                cols[2].metric("Valence", f"{row['valence']:.2f}")
                cols[3].metric("Acoustic", f"{row['acousticness']:.2f}")
                cols[4].metric("Tempo/Intensity", f"{row['tempo']:.2f}")
                st.divider()
        return

    if df.empty or vectors is None:
        st.error("Data or Vectors missing. Cannot perform search.")
        return

    with st.spinner("Searching for the perfect vibe..."):
        # Step A: Encode Text
        tokens = tokenizer.encode(query)
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            query_vector = encoder(token_tensor).cpu().numpy().astype(np.float32)
            query_vector = np.ascontiguousarray(query_vector)
            
        # Pure Numpy Search (Instantaneous for 100k, and NEVER segfaults on Mac!)
        sim_scores = np.dot(vectors, query_vector.T).flatten()
        
        # Get top 500 semantic neighbors
        top_500_idx = np.argsort(sim_scores)[-500:][::-1]
        top_500_scores = sim_scores[top_500_idx]

        if len(top_500_idx) == 0:
            st.warning("No results found.")
            return
            
        candidate_df = df.iloc[top_500_idx].copy()
        
        # Step C: Re-ranking
        candidate_df['cosine_sim'] = top_500_scores
        
        # features array
        feat_arr = candidate_df[['energy', 'danceability', 'valence', 'acousticness', 'speechiness', 'tempo']].values
        # slider weights array
        weights = np.array([slider_energy, slider_dance, slider_mood, slider_acoustic, slider_speech, slider_tempo])
        
        # Check if the sum operation handles correct positive/negative shifts
        # E.g. pulling valence down should favor low valence
        # Actually standard dot product of slider weights * normalized features.
        # But if the user wants -1.0 on mood to pull towards Sadness (which probably means low valence feature).
        # We need to compute a score that incorporates the matching:
        # For simplicity, dot product handles this intuitively if we shift features to center around 0
        # However, features are [0,1], sliders are [-1, 1].
        # Slider * Feature:
        # If Valence = 1, Slider = 1 -> contribution 1
        # If Valence = 1, Slider = -1 -> contribution -1 (pushes down)
        # If Valence = 0, Slider = -1 -> contribution 0 (neutral, should it boost? We shift feature to [-0.5, 0.5])
        
        shifted_features = feat_arr - 0.5
        slider_contributions = np.sum(shifted_features * weights, axis=1)
        
        candidate_df['final_score'] = (0.7 * candidate_df['cosine_sim']) + (0.05 * slider_contributions)
        
        # Step D: Sort & Display Top 10
        top_10 = candidate_df.sort_values(by='final_score', ascending=False).head(10)

    st.markdown(f"### Results for *'{query}'*")
    for idx, row in top_10.iterrows():
        score = row['final_score']
        sim = row['cosine_sim']
        st.markdown(f"**{row['name']}** by *{row['artists']}* (Album: {row['album']}) - Score: `{score:.3f}` *(Sim: {sim:.3f})*")
        cols = st.columns(5)
        cols[0].metric("Energy", f"{row['energy']:.2f}")
        cols[1].metric("Dance", f"{row['danceability']:.2f}")
        cols[2].metric("Valence", f"{row['valence']:.2f}")
        cols[3].metric("Acoustic", f"{row['acousticness']:.2f}")
        cols[4].metric("Tempo/Intensity", f"{row['tempo']:.2f}")
        st.divider()
        
if __name__ == "__main__":
    main()
