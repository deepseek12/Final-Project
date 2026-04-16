# Author: proud cheetah group
import pandas as pd
import os

def reduce_dataset(input_csv="data/tracks_features.csv", output_csv="data/tracks_features_small.csv"):
    print(f"Loading massive dataset from {input_csv}...")
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    df = pd.read_csv(input_csv)
    print(f"Original size: {len(df)} songs")
    
    # Filter out missing records
    df = df.dropna(subset=['artists', 'name'])
    
    # Limit each artist to maximum 3 songs to increase diversity
    print("Limiting to max 3 songs per artist to maximize diversity...")
    # .head(3) on grouping is extremely fast and deterministic
    df_small = df.groupby('artists').head(3)
    
    print(f"Size after limiting per-artist: {len(df_small)} songs")
    
    # If we still have more than 100k, randomly sample down to exactly 100k
    target_size = 100000
    if len(df_small) > target_size:
        print(f"Randomly selecting exactly {target_size} diverse tracks...")
        df_small = df_small.sample(n=target_size, random_state=42)
    
    # Save the compressed dataset
    print(f"Saving to {output_csv}...")
    df_small.to_csv(output_csv, index=False)
    print("Done! You can now run train.py which will be significantly faster!")

if __name__ == "__main__":
    reduce_dataset()
