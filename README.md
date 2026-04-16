# EchoVibe 🎵
**Hybrid Semantic-Search Music Engine**

EchoVibe is a custom PyTorch-based machine learning music recommendation engine built from scratch. It merges a deterministic multi-feature filtering system with a Transformer-backed semantic search to dynamically locate the perfect "vibe" across hundreds of thousands of tracks.

## Features
- **Custom Transformer Architecture:** Implemented entirely from scratch using `torch.nn`, fully avoiding heavy third-party wrapper libraries.
- **Contrastive Learning:** Translates natural language text inputs into a 128-dimensional latent space to perfectly align with audio feature embeddings using an InfoNCE contrastive loss function.
- **Fast Similarity Search:** Powered by a pure, contiguous NumPy matrix dot-product engine, providing robust sub-millisecond retrieval with absolute stability across all CPU architectures (including Apple Silicon `faiss` OpenMP bug avoidance).
- **Dynamic 6-Pillar Tuning:** A beautifully responsive Streamlit frontend featuring real-time visual UI metrics and instant slider re-ranking based on:
    - Vibe Energy
    - Danceability
    - Mood (Valence)
    - Acoustic Feel
    - Vocal Focus (Speechiness)
    - Intensity (Tempo)

## Setup & Running Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Compress the Dataset:**
   Because 1.2 million songs can be sluggish to test locally on a CPU, securely downsample your datasets to a diverse 100k distribution:
   ```bash
   python pipeline/reduce_dataset.py
   ```

3. **Train the Neural Networks:**
   Pre-processes the data, compiles it to an optimized Parquet file, and runs the PyTorch Autoencoder + Transformer contrastive training block.
   ```bash
   python pipeline/train.py
   ```

4. **Generate the Vector Embeddings:**
   Push all songs through the trained Autoencoder and compile the 128-dimensional latent matrix to `song_vectors.npy`.
   ```bash
   python pipeline/index_builder.py
   ```

5. **Launch the Streamlit Engine:**
   ```bash
   streamlit run app.py
   ```

## Author
**proud cheetah group**
