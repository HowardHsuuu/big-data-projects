import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

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
    results = []
    
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
                        
                        results.append({
                            'init_type': f'visual-guided-{cov_type}',
                            'random_state': seed,
                            'log_likelihood': score,
                            'converged': gmm.converged_,
                            'n_iter': gmm.n_iter_
                        })
                        
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
                    
                    results.append({
                        'init_type': f'power-transform-{cov_type}',
                        'random_state': seed,
                        'log_likelihood': score,
                        'converged': gmm.converged_,
                        'n_iter': gmm.n_iter_
                    })
                    
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
                        
                        results.append({
                            'init_type': f'pca-{n_components}',
                            'random_state': seed,
                            'log_likelihood': score,
                            'converged': gmm.converged_,
                            'n_iter': gmm.n_iter_
                        })
                        
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
                    
                    results.append({
                        'init_type': f'original-{init_type}',
                        'random_state': rs,
                        'log_likelihood': score,
                        'converged': gmm.converged_,
                        'n_iter': gmm.n_iter_
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_gmm = gmm
                        best_method = f"Original-{init_type}"
                        
                except Exception:
                    continue
    except Exception:
        pass
    
    return best_gmm, best_method, best_score, results

def plot_dimensional_relationships(X, feature_cols, labels=None, save_path=None):
    n_features = len(feature_cols)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dimensional Relationships Analysis', fontsize=16, fontweight='bold')
    
    key_pairs = [(0, 1), (1, 2), (2, 3)]
    
    for idx, (i, j) in enumerate(key_pairs):
        if i < n_features and j < n_features:
            ax = axes[0, idx]
            if labels is not None:
                scatter = ax.scatter(X[:, i], X[:, j], c=labels, alpha=0.6, s=20, cmap='tab20')
                ax.set_title(f'Clustered: {feature_cols[i]} vs {feature_cols[j]}')
            else:
                ax.scatter(X[:, i], X[:, j], alpha=0.6, s=20, color='blue')
                ax.set_title(f'Raw Data: {feature_cols[i]} vs {feature_cols[j]}')
            
            ax.set_xlabel(f'Feature {feature_cols[i]}')
            ax.set_ylabel(f'Feature {feature_cols[j]}')
            ax.grid(True, alpha=0.3)
    
    if n_features >= 6:
        additional_pairs = [(3, 4), (4, 5), (0, 5)]
        for idx, (i, j) in enumerate(additional_pairs):
            ax = axes[1, idx]
            if labels is not None:
                scatter = ax.scatter(X[:, i], X[:, j], c=labels, alpha=0.6, s=20, cmap='tab20')
                ax.set_title(f'Clustered: {feature_cols[i]} vs {feature_cols[j]}')
            else:
                ax.scatter(X[:, i], X[:, j], alpha=0.6, s=20, color='blue')
                ax.set_title(f'Raw Data: {feature_cols[i]} vs {feature_cols[j]}')
            
            ax.set_xlabel(f'Feature {feature_cols[i]}')
            ax.set_ylabel(f'Feature {feature_cols[j]}')
            ax.grid(True, alpha=0.3)
    else:
        ax = axes[1, 0]
        corr_matrix = np.corrcoef(X.T)
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Matrix')
        ax.set_xticks(range(n_features))
        ax.set_yticks(range(n_features))
        ax.set_xticklabels(feature_cols)
        ax.set_yticklabels(feature_cols)
        
        for i in range(n_features):
            for j in range(n_features):
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                       ha='center', va='center', color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax)
        
        axes[1, 1].remove()
        axes[1, 2].remove()
        ax_dist = fig.add_subplot(2, 3, (5, 6))
        for i, col in enumerate(feature_cols):
            ax_dist.hist(X[:, i], alpha=0.7, bins=30, label=f'Feature {col}', density=True)
        ax_dist.set_title('Feature Distributions')
        ax_dist.set_xlabel('Standardized Value')
        ax_dist.set_ylabel('Density')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dimensional relationships plot saved to {save_path}")
    
    plt.show()

def plot_clustering_results(X, labels, feature_cols, n_clusters, save_path=None):
    fig = plt.figure(figsize=(16, 12))
    
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1])
    
    key_pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]
    
    for idx, (i, j) in enumerate(key_pairs[:4]):
        if i < len(feature_cols) and j < len(feature_cols):
            ax = fig.add_subplot(gs[idx//2, idx%2])
            scatter = ax.scatter(X[:, i], X[:, j], c=labels, alpha=0.7, s=30, cmap='tab20')
            ax.set_xlabel(f'Feature {feature_cols[i]}')
            ax.set_ylabel(f'Feature {feature_cols[j]}')
            ax.set_title(f'{feature_cols[i]} vs {feature_cols[j]} - {n_clusters} Clusters')
            ax.grid(True, alpha=0.3)
    
    ax_bar = fig.add_subplot(gs[0, 2])
    cluster_counts = np.bincount(labels)
    bars = ax_bar.bar(range(len(cluster_counts)), cluster_counts, color='skyblue', edgecolor='navy')
    ax_bar.set_xlabel('Cluster ID')
    ax_bar.set_ylabel('Number of Points')
    ax_bar.set_title('Cluster Size Distribution')
    ax_bar.grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(cluster_counts),
                   f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    if len(feature_cols) > 2:
        ax_pca = fig.add_subplot(gs[1, 2])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.7, s=30, cmap='tab20')
        ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax_pca.set_title('PCA Projection of Clusters')
        ax_pca.grid(True, alpha=0.3)
    
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('tight')
    ax_stats.axis('off')
    
    stats_data = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_data = X[mask]
        if len(cluster_data) > 0:
            stats_data.append([
                cluster_id,
                len(cluster_data),
                f"{len(cluster_data)/len(X)*100:.1f}%",
                f"{np.mean(cluster_data, axis=0).round(2)}".replace('\n', ''),
                f"{np.std(cluster_data, axis=0).round(2)}".replace('\n', '')
            ])
    
    table = ax_stats.table(cellText=stats_data,
                          colLabels=['Cluster', 'Size', 'Percentage', 'Mean', 'Std'],
                          cellLoc='center',
                          loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax_stats.set_title('Cluster Statistics Summary', fontweight='bold', pad=20)
    
    plt.suptitle(f'Visual-Guided Clustering Results: {n_clusters} Clusters Identified', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Clustering results plot saved to {save_path}")
    
    plt.show()

def plot_algorithm_performance(results_list, dataset_names, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Enhanced Algorithm Performance Analysis', fontsize=16, fontweight='bold')
    
    all_results = []
    for i, results in enumerate(results_list):
        for result in results:
            result['dataset'] = dataset_names[i]
            all_results.append(result)
    
    df_results = pd.DataFrame(all_results)
    
    ax1 = axes[0, 0]
    init_types = df_results['init_type'].unique()
    for init_type in init_types:
        data = df_results[df_results['init_type'] == init_type]['log_likelihood']
        ax1.hist(data, alpha=0.7, label=f'{init_type}', bins=10)
    ax1.set_xlabel('Log-likelihood')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Performance by Strategy Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    converged_data = df_results[df_results['converged']]['log_likelihood']
    not_converged_data = df_results[~df_results['converged']]['log_likelihood']
    
    ax2.hist(converged_data, alpha=0.7, label='Converged', bins=10, color='green')
    ax2.hist(not_converged_data, alpha=0.7, label='Not Converged', bins=10, color='red')
    ax2.set_xlabel('Log-likelihood')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Convergence Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.hist(df_results['n_iter'], bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Number of Iterations')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Convergence Speed Distribution')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    for dataset in dataset_names:
        data = df_results[df_results['dataset'] == dataset]['log_likelihood']
        ax4.boxplot(data, positions=[dataset_names.index(dataset)], widths=0.6, 
                   patch_artist=True, boxprops=dict(facecolor=f'C{dataset_names.index(dataset)}'))
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Log-likelihood')
    ax4.set_title('Performance Comparison by Dataset')
    ax4.set_xticks(range(len(dataset_names)))
    ax4.set_xticklabels(dataset_names)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Algorithm performance plot saved to {save_path}")
    
    plt.show()

def plot_feature_correlation(X, feature_cols, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    corr_matrix = np.corrcoef(X.T)
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('Feature Correlation Matrix')
    ax1.set_xticks(range(len(feature_cols)))
    ax1.set_yticks(range(len(feature_cols)))
    ax1.set_xticklabels(feature_cols)
    ax1.set_yticklabels(feature_cols)
    
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            ax1.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                   ha='center', va='center', 
                   color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax1)
    
    ax2 = axes[1]
    for i, col in enumerate(feature_cols):
        ax2.hist(X[:, i], alpha=0.7, bins=30, label=f'Feature {col}', density=True)
    ax2.set_title('Standardized Feature Distributions')
    ax2.set_xlabel('Standardized Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Enhanced Feature Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature correlation plot saved to {save_path}")
    
    plt.show()

def cluster_dataset_enhanced(filepath, feature_cols, n_clusters, output_path, dataset_name="Dataset"):
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}: {filepath}")
    print(f"{'='*50}")
    
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} samples with {len(feature_cols)} features")
    
    X_enhanced, X_scaled = enhanced_feature_engineering(data, feature_cols)
    print(f"Enhanced features created: {X_enhanced.shape}")
    
    print("\nGenerating enhanced feature correlation analysis...")
    plot_feature_correlation(X_scaled, feature_cols, 
                           save_path=f"{dataset_name.lower()}_enhanced_feature_correlation.png")
    
    print("Analyzing dimensional relationships...")
    plot_dimensional_relationships(X_scaled, feature_cols, labels=None,
                                 save_path=f"{dataset_name.lower()}_enhanced_dimensional_relationships.png")
    
    print(f"\nFitting enhanced GMM with {n_clusters} clusters...")
    gmm, method, score, results = advanced_gmm_clustering(X_enhanced, X_scaled, n_clusters)
    
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
    
    print("\nGenerating enhanced clustering visualizations...")
    plot_clustering_results(X_scaled, labels, feature_cols, n_clusters,
                           save_path=f"{dataset_name.lower()}_enhanced_clustering_results.png")
    
    plot_dimensional_relationships(X_scaled, feature_cols, labels=labels,
                                 save_path=f"{dataset_name.lower()}_enhanced_clustered_relationships.png")
    
    output_df = pd.DataFrame({
        'id': range(len(data)),
        'label': labels
    })
    
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return labels, method, results

def main():
    print("🚀 ENHANCED Particle Accelerator Data Clustering")
    print("=" * 70)
    print("Using visual hints and advanced feature engineering\n")
    
    all_results = []
    dataset_names = []
    
    try:
        print("\n🔍 PHASE 1: Enhanced Public Dataset Analysis")
        public_labels, public_method, public_results = cluster_dataset_enhanced(
            filepath="public_data.csv",
            feature_cols=['1', '2', '3', '4'],
            n_clusters=15,
            output_path="public_submission.csv",
            dataset_name="Public"
        )
        print(f"✅ Enhanced public dataset clustering completed using {public_method}")
        all_results.append(public_results)
        dataset_names.append("Public (4D)")
        
    except Exception as e:
        print(f"❌ Enhanced public dataset failed: {e}")
    
    try:
        print("\n🔍 PHASE 2: Enhanced Private Dataset Analysis")
        private_labels, private_method, private_results = cluster_dataset_enhanced(
            filepath="private_data.csv",
            feature_cols=['1', '2', '3', '4', '5', '6'],
            n_clusters=23,
            output_path="private_submission.csv",
            dataset_name="Private"
        )
        print(f"✅ Enhanced private dataset clustering completed using {private_method}")
        all_results.append(private_results)
        dataset_names.append("Private (6D)")
        
    except Exception as e:
        print(f"❌ Enhanced private dataset failed: {e}")
    
    if len(all_results) > 0:
        print("\n📊 PHASE 3: Enhanced Comparative Performance Analysis")
        plot_algorithm_performance(all_results, dataset_names,
                                 save_path="enhanced_algorithm_performance_comparison.png")
    
    print("\n" + "="*70)
    print("🎉 Enhanced Clustering Pipeline Complete!")
    print("Generated enhanced visualizations:")
    print("  • Enhanced feature correlation matrices")
    print("  • Advanced dimensional relationship plots") 
    print("  • Visual-guided clustering result visualizations")
    print("  • Enhanced algorithm performance analysis")
    print("="*70)

if __name__ == "__main__":
    main()