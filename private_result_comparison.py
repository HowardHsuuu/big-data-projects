import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_clustering_results(file1, file2):
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        print(f"Loaded {file1}: {len(df1)} samples")
        print(f"Loaded {file2}: {len(df2)} samples")
        
        if len(df1) != len(df2):
            raise ValueError("Files have different number of samples!")
        
        if not all(df1['id'] == df2['id']):
            raise ValueError("Sample IDs don't match between files!")
            
        return df1['label'].values, df2['label'].values
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

def calculate_similarity_metrics(labels1, labels2, name1="Method 1", name2="Method 2"):
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    fmi = fowlkes_mallows_score(labels1, labels2)
    
    exact_agreement = np.mean(labels1 == labels2)
    
    n_clusters1 = len(np.unique(labels1))
    n_clusters2 = len(np.unique(labels2))
    
    print(f"\n{'='*60}")
    print(f"CLUSTERING SIMILARITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"Samples: {len(labels1):,}")
    print(f"\nCLUSTER COUNTS:")
    print(f"  {name1}: {n_clusters1} clusters")
    print(f"  {name2}: {n_clusters2} clusters")
    
    print(f"\nSIMILARITY METRICS:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"  Fowlkes-Mallows Index (FMI): {fmi:.4f}")
    print(f"  Exact Label Agreement: {exact_agreement:.1%}")
    
    print(f"\nINTERPRETATION:")
    if ari > 0.8:
        print(f"  🟢 Very High Similarity (ARI > 0.8)")
    elif ari > 0.6:
        print(f"  🟡 High Similarity (ARI > 0.6)")
    elif ari > 0.4:
        print(f"  🟠 Moderate Similarity (ARI > 0.4)")
    elif ari > 0.2:
        print(f"  🔴 Low Similarity (ARI > 0.2)")
    else:
        print(f"  ⚫ Very Low Similarity (ARI ≤ 0.2)")
    
    return {
        'ari': ari,
        'nmi': nmi,
        'fmi': fmi,
        'exact_agreement': exact_agreement,
        'n_clusters1': n_clusters1,
        'n_clusters2': n_clusters2
    }

def analyze_cluster_size_differences(labels1, labels2, name1="Method 1", name2="Method 2"):
    sizes1 = np.bincount(labels1)
    sizes2 = np.bincount(labels2)
    
    print(f"\nCLUSTER SIZE ANALYSIS:")
    print(f"\n{name1} cluster sizes:")
    for i, size in enumerate(sizes1):
        print(f"  Cluster {i}: {size:,} samples ({size/len(labels1)*100:.1f}%)")
    
    print(f"\n{name2} cluster sizes:")
    for i, size in enumerate(sizes2):
        print(f"  Cluster {i}: {size:,} samples ({size/len(labels2)*100:.1f}%)")
    
    print(f"\nSIZE DISTRIBUTION STATS:")
    print(f"  {name1} - Mean: {np.mean(sizes1):.0f}, Std: {np.std(sizes1):.0f}")
    print(f"  {name2} - Mean: {np.mean(sizes2):.0f}, Std: {np.std(sizes2):.0f}")
    
    return sizes1, sizes2

def create_comparison_visualizations(labels1, labels2, sizes1, sizes2, 
                                   name1="Method 1", name2="Method 2", save_prefix="comparison"):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1.5])
    
    ax1 = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(sizes1))
    width = 0.35
    
    ax1.bar(x_pos - width/2, sizes1, width, label=name1, alpha=0.8, color='skyblue')
    ax1.bar(x_pos + width/2, sizes2, width, label=name2, alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Cluster Size')
    ax1.set_title('Cluster Size Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(sizes1, bins=10, alpha=0.7, label=name1, color='skyblue', density=True)
    ax2.hist(sizes2, bins=10, alpha=0.7, label=name2, color='lightcoral', density=True)
    ax2.set_xlabel('Cluster Size')
    ax2.set_ylabel('Density')
    ax2.set_title('Cluster Size Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    
    sample_size = min(1000, len(labels1))
    indices = np.random.choice(len(labels1), sample_size, replace=False)
    
    agreement = (labels1[indices] == labels2[indices]).astype(int)
    agreement_by_cluster = []
    
    for cluster_id in range(max(max(labels1), max(labels2)) + 1):
        mask1 = labels1[indices] == cluster_id
        mask2 = labels2[indices] == cluster_id
        if np.sum(mask1) > 0 or np.sum(mask2) > 0:
            cluster_agreement = np.mean(agreement[mask1 | mask2]) if np.sum(mask1 | mask2) > 0 else 0
            agreement_by_cluster.append(cluster_agreement)
    
    ax3.bar(range(len(agreement_by_cluster)), agreement_by_cluster, color='green', alpha=0.7)
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Agreement Rate')
    ax3.set_title('Per-Cluster Agreement')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    scatter_indices = np.random.choice(len(labels1), min(2000, len(labels1)), replace=False)
    ax4.scatter(labels1[scatter_indices], labels2[scatter_indices], alpha=0.6, s=20)
    ax4.set_xlabel(f'{name1} Cluster Labels')
    ax4.set_ylabel(f'{name2} Cluster Labels')
    ax4.set_title('Label Assignment Comparison')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    
    max_clusters = min(15, max(max(labels1), max(labels2)) + 1)
    conf_matrix = np.zeros((max_clusters, max_clusters))
    
    for i in range(max_clusters):
        for j in range(max_clusters):
            mask1 = labels1 == i
            mask2 = labels2 == j
            conf_matrix[i, j] = np.sum(mask1 & mask2)
    
    conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    im = ax5.imshow(conf_matrix_norm, cmap='Blues', aspect='auto')
    ax5.set_xlabel(f'{name2} Clusters')
    ax5.set_ylabel(f'{name1} Clusters')
    ax5.set_title('Cluster Assignment Overlap')
    plt.colorbar(im, ax=ax5)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    fmi = fowlkes_mallows_score(labels1, labels2)
    exact_agreement = np.mean(labels1 == labels2)
    
    metrics_text = f"""
SIMILARITY METRICS

Adjusted Rand Index: {ari:.3f}
Normalized Mutual Info: {nmi:.3f}
Fowlkes-Mallows Index: {fmi:.3f}
Exact Agreement: {exact_agreement:.1%}

CLUSTER COUNTS
{name1}: {len(np.unique(labels1))} clusters
{name2}: {len(np.unique(labels2))} clusters

INTERPRETATION
{"Very High" if ari > 0.8 else "High" if ari > 0.6 else "Moderate" if ari > 0.4 else "Low" if ari > 0.2 else "Very Low"} Similarity
    """
    
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    stats_data = [
        ['Total Samples', f'{len(labels1):,}', f'{len(labels2):,}', 'Same' if len(labels1) == len(labels2) else 'Different'],
        ['Number of Clusters', f'{len(np.unique(labels1))}', f'{len(np.unique(labels2))}', 'Same' if len(np.unique(labels1)) == len(np.unique(labels2)) else 'Different'],
        ['Largest Cluster', f'{np.max(sizes1):,} ({np.max(sizes1)/len(labels1)*100:.1f}%)', f'{np.max(sizes2):,} ({np.max(sizes2)/len(labels2)*100:.1f}%)', ''],
        ['Smallest Cluster', f'{np.min(sizes1):,} ({np.min(sizes1)/len(labels1)*100:.1f}%)', f'{np.min(sizes2):,} ({np.min(sizes2)/len(labels2)*100:.1f}%)', ''],
        ['Std Deviation', f'{np.std(sizes1):.0f}', f'{np.std(sizes2):.0f}', f'Δ{abs(np.std(sizes1) - np.std(sizes2)):.0f}'],
    ]
    
    table = ax7.table(cellText=stats_data,
                     colLabels=['Metric', name1, name2, 'Comparison'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax7.set_title('Statistical Comparison Summary', fontweight='bold', pad=20)
    
    plt.suptitle(f'Clustering Comparison: {name1} vs {name2}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = f"{save_prefix}_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison visualization saved to {save_path}")
    
    plt.show()

def main():
    print("🔍 CLUSTERING RESULTS COMPARISON TOOL")
    print("=" * 60)
    
    file1 = "b11902080_private.csv"
    file2 = "private_submission.csv"
    
    name1 = "Multi-Strategy GMM"
    name2 = "Visual-Guided"
    
    print(f"Comparing clustering results:")
    print(f"  File 1: {file1} ({name1})")
    print(f"  File 2: {file2} ({name2})")
    
    labels1, labels2 = load_clustering_results(file1, file2)
    
    if labels1 is None or labels2 is None:
        print("❌ Failed to load clustering results!")
        return
    
    metrics = calculate_similarity_metrics(labels1, labels2, name1, name2)
    
    sizes1, sizes2 = analyze_cluster_size_differences(labels1, labels2, name1, name2)
    
    create_comparison_visualizations(labels1, labels2, sizes1, sizes2, name1, name2, "private_clustering")
    
    print(f"\n{'='*60}")
    print("📊 ANALYSIS COMPLETE!")
    print("Generated comprehensive comparison analysis and visualization.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()