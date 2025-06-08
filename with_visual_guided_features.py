import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def enhanced_feature_engineering(data, feature_cols):
    X = data[feature_cols].values
    X = np.nan_to_num(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    enhanced_features = [X_scaled]
    
    if len(feature_cols) >= 4:
        s1, s2, s3, s4 = X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], X_scaled[:, 3]
        
        s1_s2_features = np.column_stack([
            s1 * s2,
            s1 / (np.abs(s2) + 1e-8),
            s1 + s2,
            s1 - s2,
        ])
        
        s2_s3_features = np.column_stack([
            s2 * s3,
            s2 / (np.abs(s3) + 1e-8),
            s2 + s3,
            s2 - s3,
            np.sqrt(s2**2 + s3**2),
            np.arctan2(s3, s2 + 1e-8),
        ])
        
        s3_s4_features = np.column_stack([
            s3 * s4,
            s3 / (np.abs(s4) + 1e-8),
            s3 + s4,
            s3 - s4,
        ])
        
        enhanced_features.extend([s1_s2_features, s2_s3_features, s3_s4_features])
        
        if len(feature_cols) >= 6:
            s5, s6 = X_scaled[:, 4], X_scaled[:, 5]
            
            additional_features = np.column_stack([
                s2 * s5,
                s3 * s6,
                s5 / (np.abs(s6) + 1e-8),
                s5 + s6,
                s5 - s6,
            ])
            enhanced_features.append(additional_features)
    
    X_enhanced = np.hstack(enhanced_features)
    
    return X_enhanced, X_scaled

def visual_guided_initialization(X_scaled, n_clusters):
    if X_scaled.shape[1] >= 3:
        s2_s3_data = X_scaled[:, 1:3]
        
        initial_gmm = GaussianMixture(n_components=5, random_state=42, max_iter=100)
        s2_s3_labels = initial_gmm.fit_predict(s2_s3_data)
        
        unique_labels = np.unique(s2_s3_labels)
        centroids = []
        
        for label in unique_labels:
            mask = s2_s3_labels == label
            if np.sum(mask) > 0:
                centroid = np.mean(X_scaled[mask], axis=0)
                centroids.append(centroid)
        
        base_centroids = np.array(centroids)
        if len(base_centroids) < n_clusters:
            additional_needed = n_clusters - len(base_centroids)
            for i in range(additional_needed):
                base_idx = i % len(base_centroids)
                noise = np.random.normal(0, 0.1, X_scaled.shape[1])
                new_centroid = base_centroids[base_idx] + noise
                centroids.append(new_centroid)
        
        return np.array(centroids[:n_clusters])
    
    return None

def advanced_gmm_clustering(X_enhanced, X_scaled, n_clusters, random_state=42):
    best_gmm = None
    best_score = -np.inf
    best_method = None
    
    try:
        guided_centroids = visual_guided_initialization(X_scaled, n_clusters)
        
        if guided_centroids is not None:
            for cov_type in ['full', 'diag', 'tied']:
                for seed in [42, 123, 456]:
                    try:
                        gmm = GaussianMixture(
                            n_components=n_clusters,
                            covariance_type=cov_type,
                            random_state=seed,
                            max_iter=300,
                            reg_covar=1e-6
                        )
                        
                        gmm.fit(X_enhanced)
                        score = gmm.score(X_enhanced)
                        
                        if score > best_score:
                            best_score = score
                            best_gmm = gmm
                            best_method = f"Visual-guided-{cov_type}"
                            
                    except Exception:
                        continue
    except Exception:
        pass
    
    try:
        pt = PowerTransformer(method='yeo-johnson')
        X_power = pt.fit_transform(X_enhanced)
        
        for cov_type in ['full', 'diag']:
            for seed in [42, 123, 456, 789]:
                try:
                    gmm = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type=cov_type,
                        init_params='kmeans',
                        random_state=seed,
                        max_iter=250,
                        reg_covar=1e-6
                    )
                    
                    gmm.fit(X_power)
                    score = gmm.score(X_power)
                    
                    if score > best_score:
                        best_score = score
                        best_gmm = gmm
                        best_method = f"Power-transform-{cov_type}"
                        
                except Exception:
                    continue
    except Exception:
        pass
    
    try:
        for n_components in [min(X_enhanced.shape[1], n_clusters*2), X_enhanced.shape[1]//2]:
            if n_components > 0 and n_components <= X_enhanced.shape[1]:
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_enhanced)
                
                for seed in [42, 123, 456]:
                    try:
                        gmm = GaussianMixture(
                            n_components=n_clusters,
                            covariance_type='full',
                            random_state=seed,
                            max_iter=200,
                            reg_covar=1e-6
                        )
                        
                        gmm.fit(X_pca)
                        score = gmm.score(X_pca)
                        
                        if score > best_score:
                            best_score = score
                            best_gmm = gmm
                            best_method = f"PCA-{n_components}"
                            
                    except Exception:
                        continue
    except Exception:
        pass
    
    try:
        for init_type in ['kmeans', 'random']:
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
                    
                    gmm.fit(X_scaled)
                    score = gmm.score(X_scaled)
                    
                    if score > best_score:
                        best_score = score
                        best_gmm = gmm
                        best_method = f"Original-{init_type}"
                        
                except Exception:
                    continue
    except Exception:
        pass
    
    return best_gmm, best_method, best_score

def cluster_dataset_enhanced(filepath, feature_cols, n_clusters, output_path):
    print(f"Processing {filepath} with enhanced methods...")
    
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} samples with {len(feature_cols)} features")
    
    X_enhanced, X_scaled = enhanced_feature_engineering(data, feature_cols)
    print(f"Enhanced features created: {X_enhanced.shape}")
    
    print(f"Fitting enhanced GMM with {n_clusters} clusters...")
    gmm, method, score = advanced_gmm_clustering(X_enhanced, X_scaled, n_clusters)
    
    if gmm is None:
        print("All enhanced methods failed, using fallback K-means")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X_scaled)
        method = "K-means-fallback"
    else:
        if 'Power-transform' in method:
            pt = PowerTransformer(method='yeo-johnson')
            X_for_prediction = pt.fit_transform(X_enhanced)
        elif 'PCA' in method:
            n_comp = int(method.split('-')[1]) if '-' in method else X_enhanced.shape[1]//2
            pca = PCA(n_components=n_comp)
            X_for_prediction = pca.fit_transform(X_enhanced)
        elif 'Original' in method:
            X_for_prediction = X_scaled
        else:
            X_for_prediction = X_enhanced
            
        labels = gmm.predict(X_for_prediction)
    
    print(f"Enhanced clustering completed using {method}")
    print(f"Score: {score:.4f}")
    print(f"Cluster distribution: {np.bincount(labels)}")
    
    output_df = pd.DataFrame({
        'id': range(len(data)),
        'label': labels
    })
    
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return labels, method

def main():
    print("=== ENHANCED Particle Accelerator Data Clustering ===")
    print("Using visual hints and advanced feature engineering\n")
    
    try:
        public_labels, public_method = cluster_dataset_enhanced(
            filepath="public_data.csv",
            feature_cols=['1', '2', '3', '4'],
            n_clusters=15,
            output_path="submission_public.csv"
        )
        print(f"✓ Public dataset clustering completed using {public_method}\n")
    except Exception as e:
        print(f"✗ Public dataset failed: {e}\n")
    
    try:
        private_labels, private_method = cluster_dataset_enhanced(
            filepath="private_data.csv",
            feature_cols=['1', '2', '3', '4', '5', '6'],
            n_clusters=23,
            output_path="submission_private.csv"
        )
        print(f"✓ Private dataset clustering completed using {private_method}\n")
    except Exception as e:
        print(f"✗ Private dataset failed: {e}\n")
    
    print("=== Enhanced Clustering Pipeline Complete ===")

if __name__ == "__main__":
    main()