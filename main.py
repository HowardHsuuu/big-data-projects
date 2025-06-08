import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(data, feature_cols):
    X = data[feature_cols].values
    X = np.nan_to_num(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def initialize_gmm_with_kmeans(X, n_clusters, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    return kmeans.cluster_centers_

def fit_gmm_clustering(X, n_clusters, random_state=42):
    best_gmm = None
    best_score = -np.inf
    init_strategies = ['kmeans', 'random']
    for init_type in init_strategies:
        for rs in [random_state, random_state + 1, random_state + 2]:
            try:
                gmm = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type='full',
                    init_params=init_type,
                    max_iter=200,
                    random_state=rs,
                    reg_covar=1e-6
                )
                
                gmm.fit(X)
                score = gmm.score(X)
                if score > best_score:
                    best_score = score
                    best_gmm = gmm
                    
            except Exception as e:
                print(f"Failed with init={init_type}, rs={rs}: {e}")
                continue
    
    return best_gmm

def cluster_dataset(filepath, feature_cols, n_clusters, output_path):
    print(f"Processing {filepath}...")
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} samples with {len(feature_cols)} features")
    X_scaled, scaler = preprocess_data(data, feature_cols)
    print(f"Data preprocessed: shape {X_scaled.shape}")
    print(f"Fitting GMM with {n_clusters} clusters...")
    gmm = fit_gmm_clustering(X_scaled, n_clusters)
    if gmm is None:
        raise Exception("All GMM initialization attempts failed")
    
    labels = gmm.predict(X_scaled)
    print(f"Clustering completed. Log-likelihood: {gmm.score(X_scaled):.2f}")
    print(f"Cluster distribution: {np.bincount(labels)}")
    output_df = pd.DataFrame({
        'id': range(len(data)),
        'label': labels
    })
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return gmm, labels

def main():
    print("=== Particle Accelerator Data Clustering ===\n")
    try:
        public_gmm, public_labels = cluster_dataset(
            filepath="public_data.csv",
            feature_cols=['1', '2', '3', '4'],
            n_clusters=15,  # 4n - 1 = 4*4 - 1 = 15
            output_path="submission_public.csv"
        )
        print("✓ Public dataset clustering completed\n")
    except Exception as e:
        print(f"✗ Public dataset failed: {e}\n")

    try:
        private_gmm, private_labels = cluster_dataset(
            filepath="private_data.csv",
            feature_cols=['1', '2', '3', '4', '5', '6'],
            n_clusters=23,  # 4n - 1 = 4*6 - 1 = 23
            output_path="submission_private.csv"
        )
        print("✓ Private dataset clustering completed\n")
    except Exception as e:
        print(f"✗ Private dataset failed: {e}\n")
    
    print("=== Clustering Pipeline Complete ===")

if __name__ == "__main__":
    main()