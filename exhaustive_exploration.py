import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

class AdvancedParticleClusterer:
    
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.best_model = None
        self.best_score = -np.inf
        self.best_method = None
        
    def advanced_preprocessing(self, X):
        preprocessing_results = []
        
        scaler1 = StandardScaler()
        X_standard = scaler1.fit_transform(X)
        preprocessing_results.append(("Standard", X_standard, scaler1))
        
        scaler2 = RobustScaler()
        X_robust = scaler2.fit_transform(X)
        preprocessing_results.append(("Robust", X_robust, scaler2))
        
        try:
            transformer = PowerTransformer(method='yeo-johnson')
            X_power = transformer.fit_transform(X)
            scaler3 = StandardScaler()
            X_power_scaled = scaler3.fit_transform(X_power)
            preprocessing_results.append(("Power+Standard", X_power_scaled, (transformer, scaler3)))
        except:
            pass
            
        if X.shape[1] >= 4:
            pca = PCA(n_components=min(X.shape[1], X.shape[0]//10))
            X_pca = pca.fit_transform(X_standard)
            preprocessing_results.append(("PCA", X_pca, (scaler1, pca)))
        
        return preprocessing_results
    
    def enhanced_gmm_clustering(self, X, method_name="GMM"):
        best_gmm = None
        best_ll = -np.inf
        
        configs = [
            {'covariance_type': 'full', 'init_params': 'kmeans', 'max_iter': 300},
            {'covariance_type': 'full', 'init_params': 'random', 'max_iter': 250},
            {'covariance_type': 'diag', 'init_params': 'kmeans', 'max_iter': 300},
            {'covariance_type': 'tied', 'init_params': 'kmeans', 'max_iter': 200},
            {'covariance_type': 'spherical', 'init_params': 'kmeans', 'max_iter': 200},
            {'covariance_type': 'full', 'init_params': 'kmeans', 'reg_covar': 1e-5, 'max_iter': 300},
            {'covariance_type': 'full', 'init_params': 'kmeans', 'reg_covar': 1e-4, 'max_iter': 250},
        ]
        
        seeds = [42, 123, 456, 789, 1337, 2024, 8888]
        
        attempt = 0
        for config in configs:
            for seed in seeds:
                attempt += 1
                if attempt > 35:
                    break
                    
                try:
                    gmm = GaussianMixture(
                        n_components=self.n_clusters,
                        random_state=seed,
                        **config
                    )
                    
                    gmm.fit(X)
                    ll = gmm.score(X)
                    
                    if ll > best_ll:
                        best_ll = ll
                        best_gmm = gmm
                        
                except Exception:
                    continue
        
        if best_gmm is not None:
            labels = best_gmm.predict(X)
            return labels, best_ll, f"{method_name}_GMM"
        return None, -np.inf, f"{method_name}_GMM"
    
    def alternative_clustering_methods(self, X, method_name="Alt"):
        results = []
        
        try:
            best_kmeans = None
            best_silhouette = -1
            
            for n_init in [20, 50, 100]:
                for seed in [42, 123, 456]:
                    kmeans = KMeans(
                        n_clusters=self.n_clusters, 
                        random_state=seed, 
                        n_init=n_init,
                        max_iter=500
                    )
                    labels = kmeans.fit_predict(X)
                    
                    if len(np.unique(labels)) == self.n_clusters:
                        sil_score = silhouette_score(X, labels)
                        if sil_score > best_silhouette:
                            best_silhouette = sil_score
                            best_kmeans = labels
            
            if best_kmeans is not None:
                results.append((best_kmeans, best_silhouette, f"{method_name}_KMeans"))
                
        except Exception:
            pass
        
        if X.shape[0] < 20000:
            try:
                spectral = SpectralClustering(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,
                    affinity='rbf'
                )
                spec_labels = spectral.fit_predict(X)
                if len(np.unique(spec_labels)) == self.n_clusters:
                    spec_sil = silhouette_score(X, spec_labels)
                    results.append((spec_labels, spec_sil, f"{method_name}_Spectral"))
            except Exception:
                pass
        
        return results
    
    def ensemble_clustering(self, X, method_name="Ensemble"):
        all_results = []
        
        gmm_labels, gmm_score, gmm_method = self.enhanced_gmm_clustering(X, method_name)
        if gmm_labels is not None:
            all_results.append((gmm_labels, gmm_score, gmm_method))
        
        alt_results = self.alternative_clustering_methods(X, method_name)
        all_results.extend(alt_results)
        
        if not all_results:
            return None, -np.inf, f"{method_name}_Failed"
        
        best_result = max(all_results, key=lambda x: x[1])
        return best_result
    
    def fit_predict(self, X):
        print(f"    🔄 Comprehensive clustering with {self.n_clusters} clusters...")
        
        preprocessing_results = self.advanced_preprocessing(X)
        
        all_clustering_results = []
        
        for prep_name, X_prep, prep_objects in preprocessing_results:
            print(f"      Trying {prep_name} preprocessing...")
            
            labels, score, method = self.ensemble_clustering(X_prep, prep_name)
            
            if labels is not None:
                all_clustering_results.append({
                    'labels': labels,
                    'score': score,
                    'method': method,
                    'preprocessing': prep_name,
                    'prep_objects': prep_objects
                })
        
        if not all_clustering_results:
            raise Exception("All comprehensive clustering strategies failed")
        
        best_result = max(all_clustering_results, key=lambda x: x['score'])
        
        print(f"      Best: {best_result['method']} with {best_result['preprocessing']} (Score: {best_result['score']:.3f})")
        
        return best_result['labels'], best_result

def comprehensive_cluster_dataset(filepath, feature_cols, n_clusters, output_path):
    print(f"\n🚀 COMPREHENSIVE PROCESSING: {filepath}")
    print("─" * 60)
    
    data = pd.read_csv(filepath)
    print(f"  📊 {len(data):,} samples, {len(feature_cols)} features")
    
    X = data[feature_cols].values
    X = np.nan_to_num(X)
    
    clusterer = AdvancedParticleClusterer(n_clusters=n_clusters)
    
    labels, best_result = clusterer.fit_predict(X)
    
    unique_clusters = len(np.unique(labels))
    cluster_sizes = np.bincount(labels)
    
    print(f"  ✅ Clustering complete:")
    print(f"    Method: {best_result['method']}")
    print(f"    Preprocessing: {best_result['preprocessing']}")
    print(f"    Found clusters: {unique_clusters}/{n_clusters}")
    print(f"    Score: {best_result['score']:.4f}")
    print(f"    Cluster sizes: {cluster_sizes}")
    
    output_df = pd.DataFrame({
        'id': range(len(data)),
        'label': labels
    })
    
    output_df.to_csv(output_path, index=False)
    print(f"  💾 Saved to {output_path}")
    
    return labels

def main():
    print("🚀 COMPREHENSIVE PARTICLE ACCELERATOR CLUSTERING")
    print("=" * 70)
    
    try:
        public_labels = comprehensive_cluster_dataset(
            "public_data.csv",
            ['1', '2', '3', '4'],
            15,
            "submission_public_comprehensive.csv"
        )
        print("✅ Comprehensive public clustering completed")
        
    except Exception as e:
        print(f"❌ Comprehensive public clustering failed: {e}")
    
    try:
        private_labels = comprehensive_cluster_dataset(
            "private_data.csv", 
            ['1', '2', '3', '4', '5', '6'],
            23,
            "submission_private_comprehensive.csv"
        )
        print("✅ Comprehensive private clustering completed")
        
    except Exception as e:
        print(f"❌ Comprehensive private clustering failed: {e}")
    
    print("\n🏆 COMPREHENSIVE CLUSTERING COMPLETE!")

if __name__ == "__main__":
    main()