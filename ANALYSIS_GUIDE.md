# Big Five Personality Inventory: Human vs LLM Comparison
## Step-by-Step Analysis Guide

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Running the Analysis](#running-the-analysis)
4. [Understanding the Results](#understanding-the-results)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

**For R:**
- R version 4.0 or higher
- RStudio (recommended)

**For Python:**
- Python 3.7 or higher
- Jupyter Notebook or any Python IDE

### Required Packages

**R packages:**
```r
install.packages(c(
  "tidyverse", "psych", "effsize", "DescTools", 
  "caret", "pROC", "transport", "philentropy", 
  "GPArotation", "lavaan"
))
```

**Python packages:**
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

---

## Data Preparation

### Expected Data Format

Your data should be in CSV format with the following structure:

| respondent_id | source | item_1 | item_2 | ... | item_50 |
|--------------|--------|--------|--------|-----|---------|
| H_001 | Human | 4 | 2 | ... | 5 |
| H_002 | Human | 3 | 3 | ... | 4 |
| G_001 | GPT-4 | 5 | 1 | ... | 5 |
| C_001 | Claude | 4 | 2 | ... | 4 |
| D_001 | Deepseek | 3 | 3 | ... | 3 |

**Key requirements:**
- `respondent_id`: Unique identifier for each respondent
- `source`: Label for the data source (e.g., "Human", "GPT-4", "Claude", "Deepseek", etc.)
  - Must be "Human" for human responses
  - Can be any string for LLM model names (flexible for any models)
- `item_1` through `item_50`: Response values on 1-5 scale
- No missing values in item columns (or handle appropriately)

### Data Validation Checklist

Before running analysis, verify:

- [ ] All item columns contain only values 1-5
- [ ] Source column contains only valid labels
- [ ] Each respondent has exactly 50 item responses
- [ ] No duplicate respondent IDs within each source group
- [ ] Sufficient sample size (recommended: 500+ per group)

---

## Running the Analysis

### Option 1: Using R

1. **Load the script:**
```r
source("big_five_comparison_analysis.R")
```

2. **Load your data:**
```r
all_data <- read.csv("your_data_file.csv")
```

3. **Preprocess the data:**
```r
all_data <- preprocess_data(all_data, reverse_items = REVERSE_ITEMS)
```

4. **Compute trait scores:**
```r
trait_scores <- compute_trait_scores(all_data, trait_structure)
```

5. **Run specific analyses:**

**Table 2 - Trait-level comparisons:**
```r
table2_gpt <- trait_level_comparison(trait_scores, "Human", "GPT-4.1")
table2_claude <- trait_level_comparison(trait_scores, "Human", "Claude")
table2_llama <- trait_level_comparison(trait_scores, "Human", "Llama")
```

**Table 3 - Item-level agreement:**
```r
table3 <- bind_rows(
  item_level_agreement(all_data, "Human", "GPT-4.1"),
  item_level_agreement(all_data, "Human", "Claude"),
  item_level_agreement(all_data, "Human", "Llama")
)
```

**Table 4 - Distributional similarity:**
```r
table4_gpt <- distributional_similarity(all_data, "Human", "GPT-4.1", trait_structure)
```

**Table 5 - Structural comparison:**
```r
struct_human <- structural_comparison(all_data, "Human", trait_structure)
struct_gpt <- structural_comparison(all_data, "GPT-4.1", trait_structure)

# Compute matrix correlation
matrix_cor_gpt <- compute_matrix_similarity(
  struct_human$cor_matrix, 
  struct_gpt$cor_matrix
)

# Compute factor congruence
data_human <- all_data %>% filter(source == "Human")
data_gpt <- all_data %>% filter(source == "GPT-4.1")
factor_congruence_gpt <- compute_factor_congruence(
  data_human, 
  data_gpt, 
  trait_structure
)
```

**Table 6 - Classification:**
```r
table6 <- bind_rows(
  classification_analysis(trait_scores, "Human", "GPT-4.1"),
  classification_analysis(trait_scores, "Human", "Claude"),
  classification_analysis(trait_scores, "Human", "Llama")
)
```

**Table 7 - Response style:**
```r
table7 <- bind_rows(
  response_style_metrics(all_data, "Human", reverse_items),
  response_style_metrics(all_data, "GPT-4.1", reverse_items),
  response_style_metrics(all_data, "Claude", reverse_items),
  response_style_metrics(all_data, "Llama", reverse_items)
)
```

**Table 8 - Item-by-item:**
```r
table8_gpt <- item_by_item_comparison(all_data, "Human", "GPT-4.1")
```

6. **Export results:**
```r
write.csv(table2_gpt, "results/table2_gpt.csv", row.names = FALSE)
write.csv(table3, "results/table3.csv", row.names = FALSE)
# ... export other tables similarly
```

### Option 2: Using Python (Flexible Multi-Model Approach)

#### For Single LLM Model:

```python
from big_five_comparison_analysis import run_full_analysis

# Run complete analysis
results = run_full_analysis(
    'data/processed_human.csv',
    'data/processed_llm.csv',
    'results'
)
```

#### For Multiple LLM Models (Recommended):

```python
from big_five_comparison_analysis import run_multi_model_analysis

# Define model file paths
model_paths = {
    'GPT-4': 'data/gpt4_responses.csv',
    'Claude': 'data/claude_responses.csv',
    'Deepseek': 'data/deepseek_responses.csv',
}

# Run analysis for all models at once
results = run_multi_model_analysis(
    'data/processed_human.csv',
    model_paths,
    'results'
)
```

**Output files will be generated for each model:**
- `table3_item_agreement_gpt-4.csv`
- `table4_distributional_gpt-4.csv`
- `table4b_item_distributions_gpt-4.csv`
- (and similarly for Claude, Deepseek, etc.)

#### For Step-by-Step Analysis:

```python
from big_five_comparison_analysis import (
    load_multiple_models, load_and_preprocess_data,
    compute_trait_scores, item_level_agreement,
    distributional_similarity, item_distributional_analysis
)
import pandas as pd

# Load multiple models
model_paths = {
    'GPT-4': 'data/gpt4_responses.csv',
    'Claude': 'data/claude_responses.csv',
}

df = load_multiple_models('data/processed_human.csv', model_paths)
trait_scores = compute_trait_scores(df)

# Run analyses for each model
for model_name in model_paths.keys():
    # Table 3: Item-level agreement
    table3 = pd.DataFrame([
        item_level_agreement(df, "Human", model_name)
    ])
    table3.to_csv(f'results/table3_item_agreement_{model_name.lower()}.csv', index=False)
    
    # Table 4: Trait-level distributional similarity
    table4 = distributional_similarity(df, "Human", model_name)
    table4.to_csv(f'results/table4_distributional_{model_name.lower()}.csv', index=False)
    
    # Table 4B: Item-level distributional analysis
    table4b = item_distributional_analysis(df, "Human", model_name)
    table4b.to_csv(f'results/table4b_item_distributions_{model_name.lower()}.csv', index=False)
```

---

## Understanding the Results

### Table 2: Trait-Level Mean Comparisons

**What it shows:** Average differences between human and LLM responses for each Big Five trait.

**Key metrics:**
- **Mean_Diff**: Positive = LLM scores higher than humans
- **Cohens_d**: Effect size (0.2=small, 0.5=medium, 0.8=large)
- **p_value**: Statistical significance (p < 0.05 = significant)

**Interpretation:**
- Large positive d values (e.g., 0.99 for Conscientiousness) indicate LLMs score substantially higher
- Large negative d values (e.g., -1.10 for Neuroticism) indicate LLMs score substantially lower
- This reveals systematic biases in LLM responses

### Table 3: Item-Level Agreement

**What it shows:** Overall similarity of response patterns at the item level.

**Key metrics:**
- **Correlation**: How well item means track each other (0-1, higher = better)
- **Mean_Abs_Diff**: Average absolute difference in item means
- **Significant_Items**: How many items show significant mean differences
- **Avg_Cohens_d**: Average effect size across all items

**Interpretation:**
- High correlation (>0.80) suggests similar overall patterns
- Many significant items indicates systematic differences
- Compare across models to see which LLM is most human-like

### Table 4: Distributional Similarity (Trait-Level)

**What it shows:** How differently LLMs and humans distribute responses across the 1-5 scale for each Big Five trait.

**Key metrics:**
- **Avg_Wasserstein**: Average Wasserstein distance (lower = more similar distributions)
- **Avg_JS_divergence**: Average Jensen-Shannon divergence (lower = more similar)
- **Avg_Chi_sq**: Average χ² statistic across items (higher = more significant difference)
- **Highest_mismatch_item**: Which item in the trait shows the biggest distributional mismatch

**Interpretation:**
- Wasserstein distance compares cumulative response distributions (CDFs)
- High values (>1.0) indicate substantially different response patterns
- Example: Conscientiousness (1.07) shows larger distributional differences than Openness (0.91)
- All traits show statistically significant differences (high χ²), meaning LLMs don't just differ in average response—they select different response categories

### Table 4B: Item-Level Distributional Analysis

**What it shows:** Individual breakdown of distributional differences for each of the 50 items.

**Key metrics:**
- **Item**: Item number (1-50)
- **Trait**: Which Big Five trait the item belongs to
- **Wasserstein_Distance**: Distribution difference for that specific item
- **JS_Divergence**: JS divergence for that item
- **Chi_Square**: χ² statistic for that item

**Interpretation:**
- Shows granular details about which specific items have problematic distributions
- Items with highest Wasserstein (>1.4): Item 9 (Neuroticism), Item 10 (Openness), Item 32 (Agreeableness)
- Items with lowest Wasserstein (<0.5): Item 50 (Openness), Item 37 (Agreeableness), Item 33 (Conscientiousness)
- Useful for identifying which items LLMs answer differently at the distribution level
- Different from Table 4 because it shows individual items, not trait averages

**Key Finding:** Even though Table 3 shows moderate correlation (0.708), Table 4/4B reveal that distributions are substantially different. LLM and human responses follow different patterns across the response scale.

### Comparing Means vs Distributions: Key Insight

**Important distinction:**
- **Table 3 (Item-level agreement)** measures **mean similarity**: Do the average responses align?
- **Tables 4 & 4B (Distributional)** measure **distribution similarity**: Do people select the same response categories?

**From your results:**
- Table 3 shows: Correlation = 0.708 (moderate pattern alignment), but 72% of items have significant mean differences
- Tables 4 & 4B show: ALL traits have substantial distributional differences (Wasserstein 0.89-1.08, χ² 80-135)

**What this means:**
- LLMs understand which traits go together (0.708 correlation)
- BUT LLMs systematically respond differently:
  - Different average scores (Table 3)
  - Different distribution shapes (Tables 4 & 4B)
  
**Real-world example:**
- Both humans and LLMs might agree "Conscientiousness is more important than Neuroticism"
- But LLMs might respond with mostly 4s and 5s, while humans vary across the full 1-5 scale
- Result: Patterns match, but response distributions don't

**What it shows:** Whether LLMs show similar internal consistency and factor structure.

**Key metrics:**
- **Mean_inter_item_cor**: Average correlation between all items
- **Cronbach's alpha**: Reliability coefficient (0-1, higher = more consistent)
- **Matrix_correlation**: Similarity of correlation matrices
- **Factor_congruence**: Similarity of factor structures

**Interpretation:**
- Higher alpha in LLMs suggests overly consistent responses
- Higher inter-item correlation may indicate lack of individual variation
- Lower matrix correlation means different item relationships

### Table 6: Classification/Separability

**What it shows:** How easily you can distinguish human from LLM responses.

**Key metrics:**
- **AUC**: Area under ROC curve (0.5=chance, 1.0=perfect separation)
- **Accuracy**: Percentage correctly classified
- **F1_score**: Balance of precision and recall
- **Mahalanobis distance**: Multivariate distance between group centroids

**Interpretation:**
- High AUC (>0.85) means LLM responses are detectably different
- If you can easily classify, LLMs aren't truly mimicking humans
- Important for validity of using LLMs as synthetic respondents

### Table 7: Response Style Metrics

**What it shows:** How LLMs use the response scale compared to humans.

**Key metrics:**
- **Acquiescence_index**: Tendency to agree (higher = more agreement)
- **Extreme_response_rate**: Frequency of 1s and 5s
- **Midpoint_response_rate**: Frequency of 3s
- **Reverse_key_inconsistency**: How well reverse items are handled
- **Social_desirability_loading**: Tendency toward socially desirable answers

**Interpretation:**
- Higher acquiescence suggests "yes-saying" bias
- Higher extreme rates may indicate polarized responses
- Lower reverse-key inconsistency suggests perfect logical consistency (unrealistic)
- High social desirability suggests presenting idealized self

### Table 8: Item-by-Item Comparison

**What it shows:** Detailed breakdown for every single item.

**Key metrics:**
- **Mean_Diff**: Difference for this specific item
- **Cohens_d**: Effect size for this item
- **p_value**: Significance for this item

**Interpretation:**
- Identifies specific problematic items
- Shows patterns (e.g., all reverse-keyed items problematic)
- Useful for revising prompts or understanding LLM biases

---

## Troubleshooting

### Common Issues

**Issue: "Package not found"**
- Solution: Install missing packages using `install.packages()` (R) or `pip install` (Python)

**Issue: "Reverse items not working correctly"**
- Solution: Verify your reverse items list matches your specific IPIP inventory version

**Issue: "Low sample size warning"**
- Solution: Ensure at least 500 respondents per group for stable estimates

**Issue: "Matrix is singular" in Mahalanobis distance**
- Solution: Check for multicollinearity; may need to use regularized covariance

**Issue: "Missing values in results"**
- Solution: Check for NA values in original data; handle with na.rm=TRUE or imputation

### Statistical Considerations

1. **Multiple comparisons:** With 50 items, consider Bonferroni correction (p < 0.001)
2. **Effect sizes matter more than p-values:** With large samples, tiny differences become significant
3. **Power analysis:** Ensure sufficient sample size for your smallest expected effect
4. **Assumption checking:** Verify normality for t-tests, especially with small samples

### Validation Checks

Before accepting results, verify:

1. **Reverse scoring worked:** Check that reverse items correlate negatively with trait scores
2. **Scale reliability:** Alphas should be reasonable (0.70-0.95 range)
3. **Logical consistency:** Results should make theoretical sense
4. **Replication:** Run analysis on split-half samples to verify stability

---

## Tips for Interpretation

### What to report in a paper

**Minimal reporting:**
- Table 2 (trait comparisons)
- Table 3 (item-level summary)
- Table 6 (separability)
- Table 7 (response styles)

**Full reporting:**
- All tables
- Key visualizations
- Supplementary materials with Table 8 (all items)

### Making comparisons across LLMs

**Using the new multi-model function (recommended):**
```python
# All models analyzed automatically
model_paths = {
    'GPT-4': 'data/gpt4_responses.csv',
    'Claude': 'data/claude_responses.csv',
    'Deepseek': 'data/deepseek_responses.csv',
}
results = run_multi_model_analysis('data/human.csv', model_paths, 'results')

# Results are organized by model in results dictionary
for model_name in results['models']:
    table3 = results['models'][model_name]['table3']
    table4 = results['models'][model_name]['table4']
    table4b = results['models'][model_name]['table4b']
```

**When comparing across LLMs:**

1. **Which has highest item-level correlation?** (Table 3: Correlation)
   - Higher = better alignment with human response patterns
   
2. **Which has smallest mean differences?** (Table 3: Mean_Abs_Diff, Avg_Cohens_d)
   - Smaller = responses closer to human average values
   
3. **Which has most similar distributions?** (Table 4: Avg_Wasserstein)
   - Lower = responses distributed more like humans across the 1-5 scale
   
4. **Which specific items are problematic?** (Table 4B: Wasserstein_Distance)
   - Identify items where each model differs from humans
   
5. **Which is hardest to distinguish from humans?** (Table 6 if implemented: AUC)
   - Higher AUC = harder to distinguish = more human-like

### Red flags

Watch out for:
- Extremely high alphas (>0.95) = too consistent
- Very low reverse-key inconsistency (<0.05) = too logical
- High social desirability (>0.40) = too agreeable
- Perfect distributions (no variance) = not realistic

---

## Python Functions Reference

### Data Loading Functions

#### `load_processed_data(human_path, llm_path)`
Loads and combines human and LLM data from two separate CSV files.
```python
df = load_processed_data('data/human.csv', 'data/llm.csv')
```

#### `load_multiple_models(human_path, model_paths)` ⭐ **NEW**
Loads human data and multiple LLM models. Flexible for any number of models.
```python
models = {
    'GPT-4': 'data/gpt4.csv',
    'Claude': 'data/claude.csv',
    'Deepseek': 'data/deepseek.csv',
}
df = load_multiple_models('data/human.csv', models)
```

### Analysis Functions

#### `item_level_agreement(df, ref_group="Human", comp_group="LLM")`
Computes item-level agreement metrics (correlation, mean difference, Cohen's d).
Returns dict with: `Correlation`, `Mean_Abs_Diff`, `Significant_Items`, `Avg_Cohens_d`

#### `distributional_similarity(df, ref_group="Human", comp_group="LLM")`
Trait-level distributional analysis comparing response distributions.
Returns DataFrame with traits as rows and: `Avg_Wasserstein`, `Avg_JS_divergence`, `Avg_Chi_sq`

#### `item_distributional_analysis(df, ref_group="Human", comp_group="LLM")` ⭐ **NEW**
Item-level distributional analysis for each individual item.
Returns DataFrame with items as rows and: `Wasserstein_Distance`, `JS_Divergence`, `Chi_Square`

### Execution Functions

#### `run_full_analysis(human_filepath, llm_filepath, output_dir="results")`
Single LLM model analysis. Returns dict with `df`, `trait_scores`, `table4`, `table4b`.

#### `run_multi_model_analysis(human_filepath, model_paths, output_dir="results")` ⭐ **NEW** (Recommended)
Multi-model analysis. Automatically compares all models against humans.
```python
model_paths = {
    'GPT-4': 'data/gpt4.csv',
    'Claude': 'data/claude.csv',
}
results = run_multi_model_analysis('data/human.csv', model_paths, 'results')
```
Returns dict with: `models` (nested results for each model), `df`, `trait_scores`

---

## Next Steps

After running these analyses:

1. **Examine specific items** with largest discrepancies (Table 8)
2. **Test prompt modifications** to reduce biases
3. **Compare different model versions** (e.g., GPT-4 vs GPT-4.1)
4. **Investigate interaction effects** (e.g., do biases vary by trait?)
5. **Conduct qualitative analysis** of free-text responses if available

---

## References

For more information on these methods:

- **Cohen's d**: Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
- **Cronbach's alpha**: Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests.
- **Wasserstein distance**: Vallender, S. S. (1974). Calculation of the Wasserstein distance between probability distributions.
- **Factor congruence**: Tucker, L. R. (1951). A method for synthesis of factor analysis studies.
- **Response styles**: Paulhus, D. L. (1991). Measurement and control of response bias.

---

## Contact & Support

If you encounter issues not covered here:
1. Check that your data format exactly matches the expected structure
2. Verify all reverse items are correctly specified for your inventory version
3. Ensure sample sizes are adequate (500+ per group recommended)
4. Review error messages carefully - they often indicate the specific problem

Good luck with your analysis!
