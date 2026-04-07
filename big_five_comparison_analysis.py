"""
Big Five Personality Inventory: Human vs LLM Comparison Analysis (Python)
===========================================================================
This script performs comprehensive comparisons between human respondents and
LLM-generated responses on a 50-item IPIP Big Five inventory
===========================================================================
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.covariance import EmpiricalCovariance
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define trait structure (which items belong to which trait)
TRAIT_STRUCTURE = {
    'Extraversion': [1, 6, 11, 16, 21, 26, 31, 36, 41, 46],
    'Agreeableness': [2, 7, 12, 17, 22, 27, 32, 37, 42, 47],
    'Conscientiousness': [3, 8, 13, 18, 23, 28, 33, 38, 43, 48],
    'Neuroticism': [4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
    'Openness': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
}

# Items that need reverse scoring
REVERSE_ITEMS = [2, 4, 6, 8, 10, 12,14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 44, 46, 49]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def cohen_d(x, y):
    """Calculate Cohen's d effect size"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + 
                                                  (ny-1)*np.std(y, ddof=1)**2) / dof)

def cronbach_alpha(items):
    """Calculate Cronbach's alpha reliability"""
    items = items.dropna()
    item_count = items.shape[1]
    variance_sum = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return (item_count / (item_count - 1)) * (1 - variance_sum / total_var)

def wasserstein_distance(x, y):
    """Calculate 1D Wasserstein distance"""
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Make same length through interpolation
    n = max(len(x), len(y))
    x_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x_sorted)
    y_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y_sorted)
    
    return np.mean(np.abs(x_interp - y_interp))

def jensen_shannon_divergence(p, q):
    """Calculate Jensen-Shannon divergence"""
    p = np.asarray(p)
    q = np.asarray(q)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    m = 0.5 * (p + q)
    
    def kl_div(x, y):
        return np.sum(np.where(x != 0, x * np.log(x / y), 0))
    
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

def mahalanobis_distance(x, y, cov):
    """Calculate Mahalanobis distance between two vectors"""
    diff = x - y
    try:
        inv_cov = np.linalg.inv(cov)
        return np.sqrt(diff @ inv_cov @ diff.T)
    except:
        return np.nan

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

def load_processed_data(human_path: str, llm_path: str) -> pd.DataFrame:
    """
    Load processed human and LLM data from separate CSV files.
    
    Maps column names from EXT1, EST1, etc. to item_1, item_2, etc.
    """
    # Column mapping: EXT1-10, EST1-10, AGR1-10, CSN1-10, OPN1-10 -> item_1 to item_50
    column_mapping = {}
    item_num = 1
    
    # EXT1-10 -> item_1 to item_10
    for i in range(1, 11):
        column_mapping[f'EXT{i}'] = f'item_{item_num}'
        item_num += 1
    
    # EST1-10 -> item_11 to item_20
    for i in range(1, 11):
        column_mapping[f'EST{i}'] = f'item_{item_num}'
        item_num += 1
    
    # AGR1-10 -> item_21 to item_30
    for i in range(1, 11):
        column_mapping[f'AGR{i}'] = f'item_{item_num}'
        item_num += 1
    
    # CSN1-10 -> item_31 to item_40
    for i in range(1, 11):
        column_mapping[f'CSN{i}'] = f'item_{item_num}'
        item_num += 1
    
    # OPN1-10 -> item_41 to item_50
    for i in range(1, 11):
        column_mapping[f'OPN{i}'] = f'item_{item_num}'
        item_num += 1
    
    # Load human data
    human_df = pd.read_csv(human_path)
    human_df = human_df.rename(columns=column_mapping)
    human_df['source'] = 'Human'
    human_df['respondent_id'] = range(1, len(human_df) + 1)
    
    # Load LLM data
    llm_df = pd.read_csv(llm_path)
    llm_df = llm_df.rename(columns=column_mapping)
    llm_df['source'] = 'LLM'
    llm_df['respondent_id'] = range(1, len(llm_df) + 1)
    
    # Combine datasets
    df = pd.concat([human_df, llm_df], ignore_index=True)
    
    return df

def load_multiple_models(human_path: str, model_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Load human and multiple LLM model responses from separate CSV files.
    
    Args:
        human_path: Path to human responses CSV
        model_paths: Dictionary of {model_name: path_to_csv}
                    Example: {'GPT-4': 'data/gpt4_responses.csv',
                              'Claude': 'data/claude_responses.csv'}
    
    Returns:
        Combined dataframe with all responses and source labels
    """
    # Column mapping: EXT1-10, EST1-10, AGR1-10, CSN1-10, OPN1-10 -> item_1 to item_50
    column_mapping = {}
    item_num = 1
    
    for i in range(1, 11):
        column_mapping[f'EXT{i}'] = f'item_{item_num}'
        item_num += 1
    for i in range(1, 11):
        column_mapping[f'EST{i}'] = f'item_{item_num}'
        item_num += 1
    for i in range(1, 11):
        column_mapping[f'AGR{i}'] = f'item_{item_num}'
        item_num += 1
    for i in range(1, 11):
        column_mapping[f'CSN{i}'] = f'item_{item_num}'
        item_num += 1
    for i in range(1, 11):
        column_mapping[f'OPN{i}'] = f'item_{item_num}'
        item_num += 1
    
    # Load human data
    human_df = pd.read_csv(human_path)
    human_df = human_df.rename(columns=column_mapping)
    human_df['source'] = 'Human'
    human_df['respondent_id'] = range(1, len(human_df) + 1)
    
    dfs = [human_df]
    
    # Load each model data
    for model_name, model_path in model_paths.items():
        model_df = pd.read_csv(model_path)
        model_df = model_df.rename(columns=column_mapping)
        model_df['source'] = model_name
        model_df['respondent_id'] = range(1, len(model_df) + 1)
        dfs.append(model_df)
    
    # Combine all datasets
    df = pd.concat(dfs, ignore_index=True)
    
    return df

def load_and_preprocess_data(human_path: str, llm_path: str) -> pd.DataFrame:
    """
    Load and preprocess data from processed human and LLM CSV files.
    
    Loads data, applies reverse scoring to specified items.
    """
    df = load_processed_data(human_path, llm_path)
    
    # Reverse score specified items
    for item in REVERSE_ITEMS:
        col = f'item_{item}'
        if col in df.columns:
            df[col] = 6 - df[col]
    
    return df

def compute_trait_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean trait scores for each respondent"""
    trait_scores = df[['respondent_id', 'source']].copy()
    
    for trait_name, items in TRAIT_STRUCTURE.items():
        item_cols = [f'item_{i}' for i in items]
        trait_scores[trait_name] = df[item_cols].mean(axis=1)
    
    return trait_scores

# ==============================================================================
# TABLE 2: TRAIT-LEVEL MEAN COMPARISONS
# ==============================================================================

def trait_level_comparison(trait_scores: pd.DataFrame, 
                           ref_group: str = "Human",
                           comp_group: str = "GPT-4.1") -> pd.DataFrame:
    """
    Compare trait-level means between human and LLM responses
    """
    traits = list(TRAIT_STRUCTURE.keys())
    results = []
    
    for trait in traits:
        human_vals = trait_scores[trait_scores['source'] == ref_group][trait].values
        llm_vals = trait_scores[trait_scores['source'] == comp_group][trait].values
        
        # t-test
        t_stat, p_val = stats.ttest_ind(llm_vals, human_vals)
        
        # Cohen's d
        d = cohen_d(llm_vals, human_vals)
        
        results.append({
            'Trait': trait,
            'Human_Mean': human_vals.mean(),
            'Human_SD': human_vals.std(),
            'LLM_Mean': llm_vals.mean(),
            'LLM_SD': llm_vals.std(),
            'Mean_Diff': llm_vals.mean() - human_vals.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
            'Cohens_d': d
        })
    
    return pd.DataFrame(results)

# ==============================================================================
# TABLE 3: ITEM-LEVEL AGREEMENT
# ==============================================================================

# def item_level_agreement(df: pd.DataFrame,
#                          ref_group: str = "Human",
#                          comp_group: str = "LLM") -> dict:
#     """
#     Compute item-level agreement metrics
#     """
#     item_cols = [f'item_{i}' for i in range(1, 51)]
    
#     # Calculate means for each group
#     human_means = df[df['source'] == ref_group][item_cols].mean().values
#     llm_means = df[df['source'] == comp_group][item_cols].mean().values
    
#     # Correlation
#     correlation = np.corrcoef(human_means, llm_means)[0, 1]
    
#     # Mean absolute difference
#     mad = np.mean(np.abs(llm_means - human_means))
    
#     # Item-by-item t-tests
#     sig_count = 0
#     cohen_d_values = []
    
#     for item_col in item_cols:
#         human_vals = df[df['source'] == ref_group][item_col].values
#         llm_vals = df[df['source'] == comp_group][item_col].values
        
#         _, p_val = stats.ttest_ind(llm_vals, human_vals)
#         if p_val < 0.05:
#             sig_count += 1
        
#         d = cohen_d(llm_vals, human_vals)
#         cohen_d_values.append(abs(d))
    
#     return {
#         'Model': comp_group,
#         'Correlation': correlation,
#         'Mean_Abs_Diff': mad,
#         'Significant_Items': f"{sig_count}/50",
#         'Avg_Cohens_d': np.mean(cohen_d_values)
#     }

# ==============================================================================
# TABLE 4: DISTRIBUTIONAL SIMILARITY TRAIT LEVEL
# ==============================================================================

# def distributional_similarity(df: pd.DataFrame,
#                               ref_group: str = "Human",
#                               comp_group: str = "LLM") -> pd.DataFrame:
#     """
#     Compute distributional similarity metrics for each trait
#     """
#     results = []
    
#     for trait_name, items in TRAIT_STRUCTURE.items():
#         item_cols = [f'item_{i}' for i in items]
        
#         wasserstein_dists = []
#         js_divs = []
#         chi_sq_stats = []
#         max_chi_item = ""
#         max_chi_val = 0
        
#         for item_col in item_cols:
#             human_vals = df[df['source'] == ref_group][item_col].dropna().values
#             llm_vals = df[df['source'] == comp_group][item_col].dropna().values
            
#             # Wasserstein distance
#             wd = wasserstein_distance(human_vals, llm_vals)
#             wasserstein_dists.append(wd)
            
#             # Jensen-Shannon divergence
#             human_dist = np.bincount(human_vals.astype(int), minlength=6)[1:6]  # 1-5 scale
#             llm_dist = np.bincount(llm_vals.astype(int), minlength=6)[1:6]
            
#             human_dist = human_dist / human_dist.sum()
#             llm_dist = llm_dist / llm_dist.sum()
            
#             js = jensen_shannon_divergence(human_dist, llm_dist)
#             js_divs.append(js)
            
#             # Chi-square test
#             contingency = pd.crosstab(
#                 pd.Series(np.concatenate([['Human']*len(human_vals), ['LLM']*len(llm_vals)])),
#                 pd.Series(np.concatenate([human_vals, llm_vals]))
#             )
#             chi2, _, _, _ = stats.chi2_contingency(contingency)
#             chi_sq_stats.append(chi2)
            
#             if chi2 > max_chi_val:
#                 max_chi_val = chi2
#                 max_chi_item = item_col
        
#         results.append({
#             'Trait': trait_name,
#             'Avg_Wasserstein': np.mean(wasserstein_dists),
#             'Avg_JS_divergence': np.mean(js_divs),
#             'Avg_Chi_sq': np.mean(chi_sq_stats),
#             'Highest_mismatch_item': max_chi_item
#         })
    
#     return pd.DataFrame(results)

# # ==============================================================================
# # TABLE 4B: ITEM-LEVEL DISTRIBUTIONAL ANALYSIS
# # ==============================================================================

# def item_distributional_analysis(df: pd.DataFrame,
#                                  ref_group: str = "Human",
#                                  comp_group: str = "LLM") -> pd.DataFrame:
#     """
#     Compute distributional similarity metrics for each individual item
#     """
#     results = []
    
#     for item_num in range(1, 51):
#         item_col = f'item_{item_num}'
        
#         # Determine trait
#         trait = None
#         for trait_name, items in TRAIT_STRUCTURE.items():
#             if item_num in items:
#                 trait = trait_name
#                 break
        
#         human_vals = df[df['source'] == ref_group][item_col].dropna().values
#         llm_vals = df[df['source'] == comp_group][item_col].dropna().values
        
#         # Calculate distribution metrics
#         wd = wasserstein_distance(human_vals, llm_vals)
        
#         # Jensen-Shannon divergence
#         human_dist = np.bincount(human_vals.astype(int), minlength=6)[1:6]
#         llm_dist = np.bincount(llm_vals.astype(int), minlength=6)[1:6]
        
#         human_dist = human_dist / human_dist.sum()
#         llm_dist = llm_dist / llm_dist.sum()
        
#         js = jensen_shannon_divergence(human_dist, llm_dist)
        
#         # Chi-square test
#         contingency = pd.crosstab(
#             pd.Series(np.concatenate([['Human']*len(human_vals), ['LLM']*len(llm_vals)])),
#             pd.Series(np.concatenate([human_vals, llm_vals]))
#         )
#         chi2, _, _, _ = stats.chi2_contingency(contingency)
        
#         results.append({
#             'Item': item_num,
#             'Trait': trait,
#             'Wasserstein_Distance': wd,
#             'JS_Divergence': js,
#             'Chi_Square': chi2
#         })
    
#     return pd.DataFrame(results)

# # ==============================================================================
# # TABLE 5: STRUCTURAL COMPARISON
# # ==============================================================================

# def structural_comparison(df: pd.DataFrame, source_label: str) -> dict:
#     """
#     Compute structural metrics for one group
#     """
#     source_data = df[df['source'] == source_label]
#     item_cols = [f'item_{i}' for i in range(1, 51)]
    
#     # Correlation matrix
#     cor_matrix = source_data[item_cols].corr()
    
#     # Mean inter-item correlation
#     mask = np.triu(np.ones_like(cor_matrix), k=1).astype(bool)
#     mean_inter_item = cor_matrix.where(mask).stack().mean()
    
#     # Cronbach's alpha for each trait
#     alphas = {}
#     for trait_name, items in TRAIT_STRUCTURE.items():
#         item_cols_trait = [f'item_{i}' for i in items]
#         alpha = cronbach_alpha(source_data[item_cols_trait])
#         alphas[trait_name] = alpha
    
#     return {
#         'results': {
#             'Source': source_label,
#             'Mean_inter_item_cor': mean_inter_item,
#             **{f'Alpha_{trait}': val for trait, val in alphas.items()}
#         },
#         'cor_matrix': cor_matrix
#     }

# def compute_matrix_similarity(human_cor: pd.DataFrame, 
#                               llm_cor: pd.DataFrame) -> float:
#     """Compute correlation between correlation matrices"""
#     mask = np.triu(np.ones_like(human_cor), k=1).astype(bool)
#     human_lower = human_cor.where(mask).stack().values
#     llm_lower = llm_cor.where(mask).stack().values
    
#     return np.corrcoef(human_lower, llm_lower)[0, 1]

# # ==============================================================================
# # TABLE 6: CLASSIFICATION/SEPARABILITY
# # ==============================================================================

def classification_analysis(trait_scores: pd.DataFrame,
                            ref_group: str = "Human",
                            comp_group: str = "GPT-4.1") -> dict:
    """
    Analyze separability of human and LLM responses using classification
    """
    # Prepare data
    traits = list(TRAIT_STRUCTURE.keys())
    class_data = trait_scores[trait_scores['source'].isin([ref_group, comp_group])].copy()
    class_data['group'] = (class_data['source'] == comp_group).astype(int)
    
    X = class_data[traits].values
    y = class_data['group'].values
    
    # Logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Mahalanobis distance
    human_traits = class_data[class_data['group'] == 0][traits].values
    llm_traits = class_data[class_data['group'] == 1][traits].values
    
    pooled_data = np.vstack([human_traits, llm_traits])
    cov_matrix = np.cov(pooled_data.T)
    
    mean_human = human_traits.mean(axis=0)
    mean_llm = llm_traits.mean(axis=0)
    
    mahal_dist = mahalanobis_distance(mean_llm, mean_human, cov_matrix)
    
    return {
        'Comparison': f"{ref_group} vs {comp_group}",
        'AUC': auc,
        'Accuracy': accuracy,
        'F1_score': f1,
        'Mean_Mahalanobis': mahal_dist
    }

# # ==============================================================================
# # TABLE 7: RESPONSE STYLE METRICS
# # ==============================================================================

# def response_style_metrics(df: pd.DataFrame, source_label: str) -> dict:
#     """
#     Compute response style metrics for one group
#     """
#     source_data = df[df['source'] == source_label]
#     item_cols = [f'item_{i}' for i in range(1, 51)]
    
#     # Acquiescence index
#     acquiescence = source_data[item_cols].mean().mean() - 3
    
#     # Extreme response rate
#     extreme_count = ((source_data[item_cols] == 1) | (source_data[item_cols] == 5)).sum().sum()
#     total_responses = source_data[item_cols].notna().sum().sum()
#     extreme_rate = extreme_count / total_responses
    
#     # Midpoint response rate
#     midpoint_count = (source_data[item_cols] == 3).sum().sum()
#     midpoint_rate = midpoint_count / total_responses
    
#     # Reverse-key inconsistency
#     inconsistencies = []
#     for trait_name, items in TRAIT_STRUCTURE.items():
#         regular_items = [i for i in items if i not in REVERSE_ITEMS]
#         reverse_items_trait = [i for i in items if i in REVERSE_ITEMS]
        
#         if len(regular_items) > 0 and len(reverse_items_trait) > 0:
#             regular_cols = [f'item_{i}' for i in regular_items]
#             reverse_cols = [f'item_{i}' for i in reverse_items_trait]
            
#             reg_means = source_data[regular_cols].mean(axis=1)
#             rev_means = source_data[reverse_cols].mean(axis=1)
            
#             inconsistencies.extend(np.abs(reg_means - rev_means).dropna().values)
    
#     reverse_inconsistency = np.mean(inconsistencies) if inconsistencies else np.nan
    
#     # Social desirability loading
#     sd_items = TRAIT_STRUCTURE['Agreeableness'] + TRAIT_STRUCTURE['Conscientiousness']
#     sd_cols = [f'item_{i}' for i in sd_items]
#     sd_score = source_data[sd_cols].mean(axis=1)
#     overall_score = source_data[item_cols].mean(axis=1)
    
#     sd_loading = sd_score.corr(overall_score)
    
#     return {
#         'Source': source_label,
#         'Acquiescence_index': acquiescence,
#         'Extreme_response_rate': extreme_rate,
#         'Midpoint_response_rate': midpoint_rate,
#         'Reverse_key_inconsistency': reverse_inconsistency,
#         'Social_desirability_loading': sd_loading
#     }

# # ==============================================================================
# # TABLE 8: ITEM-BY-ITEM COMPARISON
# # ==============================================================================

# def item_by_item_comparison(df: pd.DataFrame,
#                             ref_group: str = "Human",
#                             comp_group: str = "GPT-4.1",
#                             item_names: List[str] = None) -> pd.DataFrame:
#     """
#     Detailed item-by-item comparison
#     """
#     results = []
    
#     for item_num in range(1, 51):
#         item_col = f'item_{item_num}'
        
#         human_vals = df[df['source'] == ref_group][item_col].dropna().values
#         llm_vals = df[df['source'] == comp_group][item_col].dropna().values
        
#         # Determine trait
#         trait = None
#         for trait_name, items in TRAIT_STRUCTURE.items():
#             if item_num in items:
#                 trait = trait_name[0]  # E, A, C, N, O
#                 break
        
#         # Statistics
#         t_stat, p_val = stats.ttest_ind(llm_vals, human_vals)
#         d = cohen_d(llm_vals, human_vals)
        
#         results.append({
#             'Item_No': item_num,
#             'Item_stem': item_names[item_num-1] if item_names else f"Item {item_num}",
#             'Trait': trait,
#             'Human_Mean': human_vals.mean(),
#             'LLM_Mean': llm_vals.mean(),
#             'Mean_Diff': llm_vals.mean() - human_vals.mean(),
#             't_statistic': t_stat,
#             'p_value': p_val,
#             'Cohens_d': d
#         })
    
#     return pd.DataFrame(results)

# # ==============================================================================
# # VISUALIZATION FUNCTIONS
# # ==============================================================================

# def plot_trait_comparison(table2_results: pd.DataFrame, output_file: str = None):
#     """Plot trait-level mean differences"""
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     colors = plt.cm.RdYlBu_r(np.abs(table2_results['Cohens_d']) / 
#                               table2_results['Cohens_d'].abs().max())
    
#     bars = ax.bar(table2_results['Trait'], table2_results['Mean_Diff'], color=colors)
#     ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
#     ax.set_xlabel('Big Five Trait', fontsize=12)
#     ax.set_ylabel('Mean Difference (LLM - Human)', fontsize=12)
#     ax.set_title('Trait-Level Mean Differences: LLM vs Human', fontsize=14, fontweight='bold')
    
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
    
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.show()

# def plot_item_heatmap(table8_results: pd.DataFrame, output_file: str = None):
#     """Plot heatmap of effect sizes by item"""
#     # Pivot data for heatmap
#     pivot_data = table8_results.pivot_table(
#         values='Cohens_d', 
#         index='Trait', 
#         columns='Item_No', 
#         aggfunc='first'
#     )
    
#     fig, ax = plt.subplots(figsize=(16, 6))
#     sns.heatmap(pivot_data, cmap='RdBu_r', center=0, 
#                 cbar_kws={'label': "Cohen's d"},
#                 linewidths=0.5, ax=ax)
    
#     ax.set_title("Effect Sizes by Item and Trait", fontsize=14, fontweight='bold')
#     ax.set_xlabel('Item Number', fontsize=12)
#     ax.set_ylabel('Trait', fontsize=12)
    
#     plt.tight_layout()
    
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_multi_model_analysis(human_filepath: str, model_paths: Dict[str, str], output_dir: str = "results"):
    """
    Run complete analysis comparing human responses against multiple LLM models.
    
    Args:
        human_filepath: Path to human responses CSV
        model_paths: Dictionary of {model_name: path_to_csv}
                    Example: {'GPT-4': 'data/gpt4.csv',
                              'Claude': 'data/claude.csv',
                              'Deepseek': 'data/deepseek.csv'}
        output_dir: Directory to save results
    
    Example usage:
        results = run_multi_model_analysis(
            'data/processed_human.csv',
            {
                'GPT-4': 'data/gpt4_responses.csv',
                'Claude': 'data/claude_responses.csv',
                'Deepseek': 'data/deepseek_responses.csv'
            },
            'results'
        )
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    df = load_multiple_models(human_filepath, model_paths)
    trait_scores = compute_trait_scores(df)
    
    print(f"\nLoaded {len(df)} total responses")
    print(f"  - Human: {len(df[df['source'] == 'Human'])}")
    
    for model_name in model_paths.keys():
        count = len(df[df['source'] == model_name])
        print(f"  - {model_name}: {count}")
    
    all_results = {
        'df': df,
        'trait_scores': trait_scores,
        'models': {},
        'tables': {}
    }
    
    # ===========================================================================
    # TABLE 2: TRAIT-LEVEL MEAN COMPARISONS
    # ===========================================================================
    # Uncomment to run Table 2 analysis
    # print("\n" + "="*80)
    # print("Computing Table 2: Trait-level mean comparisons")
    # print("="*80)
    # table2_results = []
    # for model_name in model_paths.keys():
    #     traits = list(TRAIT_STRUCTURE.keys())
    #     for trait in traits:
    #         human_vals = trait_scores[trait_scores['source'] == 'Human'][trait].values
    #         model_vals = trait_scores[trait_scores['source'] == model_name][trait].values
    #         
    #         t_stat, p_val = stats.ttest_ind(model_vals, human_vals)
    #         d = cohen_d(model_vals, human_vals)
    #         
    #         table2_results.append({
    #             'Model': model_name,
    #             'Trait': trait,
    #             'Human_Mean': human_vals.mean(),
    #             'Human_SD': human_vals.std(),
    #             'Model_Mean': model_vals.mean(),
    #             'Model_SD': model_vals.std(),
    #             'Mean_Diff': model_vals.mean() - human_vals.mean(),
    #             't_statistic': t_stat,
    #             'p_value': p_val,
    #             'Cohens_d': d
    #         })
    # table2 = pd.DataFrame(table2_results)
    # table2.to_csv(f"{output_dir}/table2_trait_comparison.csv", index=False)
    # print(f"Saved: {output_dir}/table2_trait_comparison.csv")
    # all_results['tables']['table2'] = table2
    
    # ===========================================================================
    # TABLE 3: ITEM-LEVEL AGREEMENT
    # ===========================================================================
    # Uncomment to run Table 3 analysis
    # print("\n" + "="*80)
    # print("Computing Table 3: Item-level agreement")
    # print("="*80)
    # table3_results = []
    # for model_name in model_paths.keys():
    #     print(f"  - {model_name}")
    #     result = item_level_agreement(df, "Human", model_name)
    #     table3_results.append(result)
    # table3 = pd.DataFrame(table3_results)
    # table3.to_csv(f"{output_dir}/table3_item_agreement.csv", index=False)
    # print(f"Saved: {output_dir}/table3_item_agreement.csv")
    # print("\nTable 3 Results:")
    # print(table3.to_string(index=False))
    # all_results['tables']['table3'] = table3
    
    # ===========================================================================
    # TABLE 4: DISTRIBUTIONAL SIMILARITY (TRAIT-LEVEL)
    # ===========================================================================
    # Uncomment to run Table 4 analysis
    # print("\n" + "="*80)
    # print("Computing Table 4: Distributional similarity (trait-level)")
    # print("="*80)
    # table4_all_models = []
    # for model_name in model_paths.keys():
    #     print(f"  - {model_name}")
    #     table4 = distributional_similarity(df, "Human", model_name)
    #     table4['Model'] = model_name
    #     table4_all_models.append(table4)
    # table4 = pd.concat(table4_all_models, ignore_index=True)
    # table4.to_csv(f"{output_dir}/table4_distributional_similarity.csv", index=False)
    # print(f"Saved: {output_dir}/table4_distributional_similarity.csv")
    # all_results['tables']['table4'] = table4
    
    # ===========================================================================
    # TABLE 4B: ITEM-LEVEL DISTRIBUTIONAL ANALYSIS
    # ===========================================================================
    # Uncomment to run Table 4B analysis
    # print("\n" + "="*80)
    # print("Computing Table 4B: Item-level distributional analysis")
    # print("="*80)
    # table4b_all_models = []
    # for model_name in model_paths.keys():
    #     print(f"  - {model_name}")
    #     table4b = item_distributional_analysis(df, "Human", model_name)
    #     table4b['Model'] = model_name
    #     table4b_all_models.append(table4b)
    # table4b = pd.concat(table4b_all_models, ignore_index=True)
    # table4b.to_csv(f"{output_dir}/table4b_item_distributions.csv", index=False)
    # print(f"Saved: {output_dir}/table4b_item_distributions.csv")
    # all_results['tables']['table4b'] = table4b
    
    # ===========================================================================
    # TABLE 5: STRUCTURAL COMPARISON
    # ===========================================================================
    # Uncomment to run Table 5 analysis
    # print("\n" + "="*80)
    # print("Computing Table 5: Structural comparison")
    # print("="*80)
    # struct_human = structural_comparison(df, "Human")
    # struct_results = [
    #     {**struct_human['results'], 'Matrix_correlation': np.nan, 'Factor_congruence': np.nan}
    # ]
    # human_cor_matrix = struct_human['cor_matrix']
    # 
    # for model_name in model_paths.keys():
    #     print(f"  - {model_name}")
    #     struct_model = structural_comparison(df, model_name)
    #     matrix_cor = compute_matrix_similarity(human_cor_matrix, struct_model['cor_matrix'])
    #     struct_results.append({
    #         **struct_model['results'],
    #         'Matrix_correlation': matrix_cor,
    #         'Factor_congruence': np.nan
    #     })
    # 
    # table5 = pd.DataFrame(struct_results)
    # table5.to_csv(f"{output_dir}/table5_structural_comparison.csv", index=False)
    # print(f"Saved: {output_dir}/table5_structural_comparison.csv")
    # print("\nTable 5 Results Summary:")
    # for idx, row in table5.iterrows():
    #     print(f"\n  {row['Source']}:")
    #     print(f"    Mean inter-item correlation: {row['Mean_inter_item_cor']:.3f}")
    #     if pd.notna(row['Matrix_correlation']):
    #         print(f"    Matrix correlation: {row['Matrix_correlation']:.3f}")
    #     for trait in TRAIT_STRUCTURE.keys():
    #         alpha_col = f'Alpha_{trait}'
    #         if alpha_col in row and pd.notna(row[alpha_col]):
    #             print(f"    Alpha ({trait}): {row[alpha_col]:.3f}")
    # all_results['tables']['table5'] = table5
    
    # ===========================================================================
    # TABLE 6: CLASSIFICATION/SEPARABILITY
    # ===========================================================================
    print("\n" + "="*80)
    print("Computing Table 6: Classification/Separability")
    print("="*80)
    table6_results = []
    for model_name in model_paths.keys():
        print(f"  - {model_name}")
        result = classification_analysis(trait_scores, "Human", model_name)
        table6_results.append(result)
    
    table6 = pd.DataFrame(table6_results)
    table6.to_csv(f"{output_dir}/table6_classification.csv", index=False)
    print(f"Saved: {output_dir}/table6_classification.csv")
    print("\nTable 6 Results Summary (Classification/Separability):")
    print(table6.to_string(index=False))
    all_results['tables']['table6'] = table6
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! All results saved to {output_dir}/")
    print(f"{'='*80}")
    
    return all_results

def run_full_analysis(human_filepath: str, llm_filepath: str, output_dir: str = "results"):
    """
    Run complete analysis pipeline on human and LLM data
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(human_filepath, llm_filepath)
    trait_scores = compute_trait_scores(df)
    
    print(f"Loaded {len(df)} total responses")
    print(f"  - Human: {len(df[df['source'] == 'Human'])}")
    print(f"  - LLM: {len(df[df['source'] == 'LLM'])}")
    
    print("\nRunning analyses...")
    
    # Table 2: Trait-level comparisons
    # print("  - Table 2: Trait-level comparisons")
    # table2 = trait_level_comparison(trait_scores, "Human", "LLM")
    # table2.to_csv(f"{output_dir}/table2_trait_comparison_human_vs_llm.csv", index=False)
    
    # Table 3: Item-level agreement
    # print("  - Table 3: Item-level agreement")
    # table3 = pd.DataFrame([
    #     item_level_agreement(df, "Human", "LLM")
    # ])
    # table3.to_csv(f"{output_dir}/table3_item_level_agreement.csv", index=False)
    # print("\nItem-level agreement results:")
    # print(table3.to_string())
    
    # # Table 4: Distributional similarity
    # print("  - Table 4: Distributional similarity")
    # table4 = distributional_similarity(df, "Human", "LLM")
    # table4.to_csv(f"{output_dir}/table4_distributional_similarity.csv", index=False)
    # print("\nDistributional similarity results:")
    # print(table4.to_string())
    
    # # Table 4B: Item-level distributional analysis
    # print("\n  - Table 4B: Item-level distributional analysis")
    # table4b = item_distributional_analysis(df, "Human", "LLM")
    # table4b.to_csv(f"{output_dir}/table4b_item_distributions.csv", index=False)
    # print("\nItem-level distribution results:")
    # print(table4b.to_string())
    
    # # Table 5: Structural comparison
    # print("  - Table 5: Structural comparison")
    # struct_human = structural_comparison(df, "Human")
    # struct_gpt = structural_comparison(df, "GPT-4.1")
    # struct_claude = structural_comparison(df, "Claude")
    # struct_llama = structural_comparison(df, "Llama")
    
    # matrix_cor_gpt = compute_matrix_similarity(struct_human['cor_matrix'], struct_gpt['cor_matrix'])
    # matrix_cor_claude = compute_matrix_similarity(struct_human['cor_matrix'], struct_claude['cor_matrix'])
    # matrix_cor_llama = compute_matrix_similarity(struct_human['cor_matrix'], struct_llama['cor_matrix'])
    
    # table5 = pd.DataFrame([
    #     {**struct_human['results'], 'Matrix_correlation': np.nan, 'Factor_congruence': np.nan},
    #     {**struct_gpt['results'], 'Matrix_correlation': matrix_cor_gpt, 'Factor_congruence': np.nan},
    #     {**struct_claude['results'], 'Matrix_correlation': matrix_cor_claude, 'Factor_congruence': np.nan},
    #     {**struct_llama['results'], 'Matrix_correlation': matrix_cor_llama, 'Factor_congruence': np.nan}
    # ])
    # table5.to_csv(f"{output_dir}/table5_structural_comparison.csv", index=False)
    
    # # Table 6: Classification
    print("  - Table 6: Classification/separability")
    table6 = pd.DataFrame([
        classification_analysis(trait_scores, "Human", "GPT-4.1"),
        classification_analysis(trait_scores, "Human", "Claude"),
        classification_analysis(trait_scores, "Human", "Llama")
    ])
    table6.to_csv(f"{output_dir}/table6_classification.csv", index=False)
    
    # # Table 7: Response style
    # print("  - Table 7: Response style metrics")
    # table7 = pd.DataFrame([
    #     response_style_metrics(df, "Human"),
    #     response_style_metrics(df, "GPT-4.1"),
    #     response_style_metrics(df, "Claude"),
    #     response_style_metrics(df, "Llama")
    # ])
    # table7.to_csv(f"{output_dir}/table7_response_style.csv", index=False)
    
    # # Table 8: Item-by-item
    # print("  - Table 8: Item-by-item comparison")
    # table8_gpt = item_by_item_comparison(df, "Human", "GPT-4.1")
    # table8_gpt.to_csv(f"{output_dir}/table8_item_by_item_gpt.csv", index=False)
    
    # # Visualizations
    # print("\nCreating visualizations...")
    # plot_trait_comparison(table2_gpt, f"{output_dir}/trait_comparison_plot.png")
    # plot_item_heatmap(table8_gpt, f"{output_dir}/item_heatmap.png")
    
    # print(f"\nAnalysis complete! Results saved to {output_dir}/")

    return {
        'df': df,
        'trait_scores': trait_scores,
        # 'table2': None,  # table2,
        # 'table3': None,  # table3,
        # 'table4': table4,
        # 'table4b': table4b,
        # 'table 5: structural_comparison': table5,
        'table6': table6
    }


if __name__ == "__main__":
    # ========================================================================
    # OPTION 1: Single LLM Model Analysis (Original approach)
    # ========================================================================
    # results = run_full_analysis(
    #     'data/processed_human.csv',
    #     'data/processed_llm.csv',
    #     'results'
    # )
    
    # ========================================================================
    # OPTION 2: Multiple LLM Models Comparison (New flexible approach)
    # ========================================================================
    # Define your model paths here. Each key is the model name (label in output files)
    # and the value is the path to the CSV file with that model's responses
    
    model_paths = {
        'LLM': 'data/processed_llm.csv',
        # 'GPT-4': 'data/gpt4_responses.csv',
        # 'Claude': 'data/claude_responses.csv',
        # 'Deepseek': 'data/deepseek_responses.csv',
    }
    
    results = run_multi_model_analysis(
        'data/processed_human.csv',
        model_paths,
        'results'
    )
    
    print(f"\n{'='*80}")
    print("Analysis complete! Results saved to results/")
    print(f"{'='*80}")
