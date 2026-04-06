# Big Five Personality Inventory: Human vs LLM Comparison Analysis

Complete toolkit for comparing human respondents with LLM-generated responses on the IPIP Big Five personality inventory.

## 📁 Files Included

| File | Description | Use When |
|------|-------------|----------|
| `big_five_comparison_analysis.R` | Complete R analysis script | You prefer R / need advanced psychometric analyses |
| `big_five_comparison_analysis.py` | Complete Python analysis script | You prefer Python / want easier automation |
| `QUICK_START.R` | Simple template to get started | You want to run analysis quickly |
| `ANALYSIS_GUIDE.md` | Detailed documentation | You need help understanding methods/results |

## 🚀 Quick Start

### For R Users

1. **Install dependencies:**
```r
install.packages(c("tidyverse", "psych", "effsize", "DescTools", 
                   "caret", "pROC", "transport", "philentropy", 
                   "GPArotation", "lavaan"))
```

2. **Prepare your data** as a CSV file with this structure:
   - Columns: `respondent_id`, `source`, `item_1`, `item_2`, ..., `item_50`
   - `source` should be: "Human", "GPT-4.1", "Claude", or "Llama"
   - Items should be on 1-5 scale

3. **Run the analysis:**
```r
source("big_five_comparison_analysis.R")
all_data <- read.csv("your_data.csv")
all_data <- preprocess_data(all_data, reverse_items)
trait_scores <- compute_trait_scores(all_data, trait_structure)

# Run specific analyses
table2_gpt <- trait_level_comparison(trait_scores, "Human", "GPT-4.1")
```

### For Python Users

1. **Install dependencies:**
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

2. **Run complete analysis:**
```python
import big_five_comparison_analysis as bfa

# One command does everything!
results = bfa.run_full_analysis("your_data.csv", "results")
```

## 📊 What Analyses Are Included?

The toolkit reproduces all tables from the comparison document:

### Table 1: Study Design Summary
- Documents your sample sizes and data sources

### Table 2: Trait-Level Mean Comparisons
- Compares average scores on each Big Five trait
- Includes t-tests and Cohen's d effect sizes
- Shows whether LLMs score higher/lower than humans on each trait

### Table 3: Item-Level Agreement
- Correlation between human and LLM item means
- Number of items with significant differences
- Average effect size across all 50 items

### Table 4: Distributional Similarity
- Wasserstein distance (how different are the distributions?)
- Jensen-Shannon divergence (information-theoretic measure)
- Chi-square tests for each item
- Identifies items with worst mismatch

### Table 5: Structural Comparison
- Mean inter-item correlations
- Cronbach's alpha reliability for each trait
- Matrix correlation (similarity of item relationships)
- Factor congruence (similarity of factor structure)

### Table 6: Classification/Separability
- How easily can you distinguish human from LLM responses?
- ROC AUC, accuracy, F1 scores
- Mahalanobis distance between groups

### Table 7: Response Style Metrics
- Acquiescence (agreement bias)
- Extreme response rates (1s and 5s)
- Midpoint response rates (3s)
- Reverse-key inconsistency
- Social desirability loading

### Table 8: Item-by-Item Detailed Comparison
- Complete breakdown for all 50 items
- Mean differences, t-statistics, p-values, Cohen's d
- Organized by Big Five trait
- Useful for identifying specific problematic items

## 🎯 Typical Workflow

1. **Load and preprocess** your data (handles reverse scoring)
2. **Compute trait scores** for each respondent
3. **Run comparisons** between humans and each LLM
4. **Export results** to CSV files
5. **Generate visualizations** (trait comparison plots, heatmaps)
6. **Interpret findings** using the analysis guide

## 📈 Visualizations Generated

- **Trait comparison bar chart**: Shows mean differences with color-coded effect sizes
- **Item-level heatmap**: Displays effect sizes for all 50 items by trait
- Easy to customize for publication-ready figures

## ⚙️ Customization

### Modify Reverse Items

If your inventory uses different reverse-scored items:

```r
# R version
custom_reverse <- c(2, 6, 8, ...)  # Your reverse items
all_data <- preprocess_data(all_data, custom_reverse)
```

```python
# Python version
bfa.REVERSE_ITEMS = [2, 6, 8, ...]  # Your reverse items
```

### Compare Different Models

Simply change the comparison group:

```r
table2_custom <- trait_level_comparison(trait_scores, "Human", "YourModel")
```

### Add Item Names

For more readable output:

```r
item_names <- c("Am the life of the party", "Feel little concern for others", ...)
table8 <- item_by_item_comparison(all_data, "Human", "GPT-4.1", item_names)
```

## 🔍 What to Look For in Results

### Signs of Good LLM Performance
- High correlation with human item means (>0.80)
- Small Cohen's d values (<0.30)
- Low classification AUC (<0.70)
- Human-like response styles
- Similar factor structure

### Red Flags
- Very high Cronbach's alpha (>0.95) → too consistent
- Low reverse-key inconsistency (<0.05) → too logical
- High acquiescence (>0.30) → too agreeable
- High social desirability (>0.40) → unrealistically positive
- Many items with large effect sizes (|d| > 0.50)

## 📖 Interpreting Effect Sizes

**Cohen's d interpretation:**
- 0.0 - 0.2: Negligible
- 0.2 - 0.5: Small
- 0.5 - 0.8: Medium
- 0.8+: Large

**What positive/negative means:**
- Positive d: LLM scores **higher** than humans
- Negative d: LLM scores **lower** than humans

Example: d = 0.99 for Conscientiousness means the LLM scored about 1 standard deviation higher than humans on conscientiousness items.

## 🔧 Troubleshooting

### "Package not found"
Install the missing package using `install.packages()` (R) or `pip install` (Python)

### "Dimensions don't match"
Check that you have exactly 50 item columns named `item_1` through `item_50`

### "Reverse scoring not working"
Verify your reverse items match your specific inventory version

### "Results seem wrong"
- Check your data format carefully
- Verify source labels are exactly: "Human", "GPT-4.1", "Claude", "Llama"
- Ensure item responses are 1-5 (not 0-4 or other scales)
- Look for missing values

## 📚 Resources

- **Detailed guide**: See `ANALYSIS_GUIDE.md` for comprehensive documentation
- **Quick start**: See `QUICK_START.R` for a minimal working example
- **Full scripts**: See `big_five_comparison_analysis.R` or `.py` for complete implementations

## 🎓 Citation

If you use this toolkit in your research, please cite the original psychometric methods:

- **Big Five inventory**: Goldberg, L. R. (1992). The development of markers for the Big-Five factor structure.
- **IPIP**: Goldberg, L. R., et al. (2006). The International Personality Item Pool.
- **Cohen's d**: Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
- **Cronbach's alpha**: Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests.

## 💡 Tips for Publication

### Essential tables to include:
1. Table 1 (study design)
2. Table 2 (trait comparisons) - **most important**
3. Table 6 (separability)
4. Table 7 (response styles)

### Supplementary materials:
- Table 8 (all 50 items)
- Tables 3-5 (detailed psychometrics)
- Visualizations

### Key statistics to report:
- Sample sizes for each group
- Mean differences and Cohen's d for each trait
- Overall correlation between human and LLM means
- Classification accuracy
- Response style differences

## ⚠️ Important Notes

1. **Sample size**: You need substantial samples for stable estimates (500+ per group recommended)
2. **Multiple comparisons**: With 50 items, consider adjusting alpha (e.g., Bonferroni correction)
3. **Effect size emphasis**: Focus on Cohen's d, not just p-values (large samples make everything significant)
4. **Validity**: High classification accuracy suggests LLMs aren't truly mimicking human responses

## 🤝 Support

For questions or issues:

1. Check the `ANALYSIS_GUIDE.md` for detailed explanations
2. Verify your data format matches the expected structure exactly
3. Review the example code in `QUICK_START.R`
4. Ensure all reverse items are correctly specified for your inventory

## 📝 Example Data Format

```csv
respondent_id,source,item_1,item_2,item_3,item_4,item_5,...,item_50
H_001,Human,4,2,5,3,4,...,5
H_002,Human,3,3,4,2,3,...,4
H_003,Human,5,1,3,4,5,...,3
G_001,GPT-4.1,5,1,4,2,5,...,5
G_002,GPT-4.1,4,2,5,1,4,...,4
C_001,Claude,4,2,3,2,4,...,4
L_001,Llama,3,3,2,3,3,...,3
```

---

**Good luck with your analysis!** 🎉
