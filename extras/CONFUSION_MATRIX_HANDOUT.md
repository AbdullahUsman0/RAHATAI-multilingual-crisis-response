# Confusion Matrix Explanation & Talking Points

## What is a Confusion Matrix?

A **confusion matrix** is a table that shows how well a classification model performs by comparing **predicted labels** against **actual labels**. It reveals not just accuracy, but *where and how* the model makes mistakes.

---

## Example: 6-Class Crisis Classification

### The Matrix (Transformer Model Test Results)

```
Predicted →
Actual ↓     Class0  Class1  Class2  Class3  Class4  Class5
Class0          32       6       3       1       9       0
Class1           3      67       0       1       6       0
Class2           5       3      20       2      15       1
Class3           0       3       1     123      16       0
Class4           7      13       2       6     106       1
Class5           1       8       1       7       8       7
```

### How to Read It

- **Rows** = True labels (what the data actually is)
- **Columns** = Predicted labels (what the model guessed)
- **Diagonal values** = Correct predictions ✓
- **Off-diagonal values** = Misclassifications ✗

**Example:** The model correctly classified **67 Class1 samples** (diagonal), but also predicted 6 Class1 samples as Class0, and 6 as Class4.

---

## Key Metrics Derived From This Matrix

### 1. **Accuracy**
- Definition: Correct predictions / Total predictions
- Formula: Sum of diagonal / Sum of all cells
- This model: **73.3%** correct overall

### 2. **Per-Class Precision**
- Definition: How many predicted-positive are *actually* positive
- Formula: Diagonal value / Sum of column
- **Class0 Precision:** 32 / (32+3+5+0+7+1) = **71%**
  - *Talking point:* "When we predict Class0, we're right 71% of the time."

### 3. **Per-Class Recall**
- Definition: How many actual positives did we *find*
- Formula: Diagonal value / Sum of row
- **Class0 Recall:** 32 / (32+6+3+1+9+0) = **64%**
  - *Talking point:* "Of the true Class0 samples, we caught 64% of them."

### 4. **F1-Score**
- Definition: Harmonic mean of precision and recall
- Balances both metrics
- This model: **72.0%** weighted F1

---

## Common Patterns & What They Mean

### 1. **Strong Diagonal (Good Model)**
When most values are on the diagonal, the model is correctly distinguishing classes.
- **Our model:** Class3 (123/143 = 86% recall) is learned very well.

### 2. **High Off-Diagonal in One Row (Confused Class)**
- **Our model:** Class0 is confused with Class4 (9 misclassifications).
  - *Talking point:* "Class0 and Class4 samples share similar features; we should review those training examples."

### 3. **High Off-Diagonal in One Column (Overactive Class)**
- The model predicts this class too often for other classes.
- **Our model:** Class3 gets predicted a lot; column sum = 143 vs Class0 row sum = 61.
  - *Talking point:* "Class3 may be overrepresented in training or have broad feature overlap."

---

## How to Present to Your TA

### Opening Statement
> "This is a confusion matrix from our Transformer classifier on 163 test samples across 6 crisis classes. It shows every prediction the model made, organized by true label and predicted label."

### Key Findings

1. **Overall Performance:** 73.3% accuracy is strong for a 6-class problem; baseline random guessing would be ~16.7%.

2. **Best-Performing Class:** 
   - Class3 (likely "evacuation" or similar) with 123/143 correct (86% recall). 
   - *Why:* Probably has distinctive keywords or clear training examples.

3. **Most Confused Pair:**
   - Class0 and Class4 are frequently mixed up (9 misclassifications in either direction).
   - *Next step:* Review the training data; these classes may need better feature separation or re-labeling.

4. **Class5 (Weakest):**
   - Only 7/32 correct (22% recall). The model rarely predicts this class.
   - *Possible reason:* Under-represented in training data, or features too similar to other classes.
   - *Recommendation:* Collect more Class5 examples or apply class-weighting during training.

### Closing Statement
> "The diagonal is strong, which means the model learned most classes well. The off-diagonal tells us where to focus: more data for Class5, and feature engineering to separate Class0 from Class4."

---

## Quick Reference: Reading the Numbers

| Metric | Formula | Our Model | Interpretation |
|--------|---------|-----------|-----------------|
| **Accuracy** | Correct / Total | 73.3% | How often right overall |
| **Precision** | True Pos / Predicted Pos | 74.3% | When we say "Class X," how right are we? |
| **Recall** | True Pos / Actual Pos | 73.3% | Of all true Class X, how many did we find? |
| **F1** | 2 × (P × R)/(P + R) | 72.0% | Balanced score (precision + recall) |
| **Top-3 Accuracy** | Correct in top 3 predictions | 94.0% | If we give 3 guesses, how often are we right? |

---

## Visual Summary

```
✓ Strong Points:
  • Class3 very well learned (86% recall)
  • High top-3 accuracy (94%) means model is often "close"
  • Weighted F1 (72%) is decent across all classes

⚠ Areas to Improve:
  • Class5 severely under-learned (22% recall)
  • Class0↔Class4 confusion needs investigation
  • Recall varies widely (22% to 86%) — some classes harder than others
```

---

## Questions TA Might Ask (& Your Answers)

**Q: "Why is accuracy (73.3%) higher than recall (73.3%)?"**  
A: They're the same in this case due to weighted averaging. But per-class recall varies (Class3: 86%, Class5: 22%), showing uneven learning.

**Q: "How do you know Class0 and Class4 are related?"**  
A: The confusion matrix shows 9 samples labeled Class0 predicted as Class4, and 7 vice versa. We'd examine those specific samples to find shared keywords or features.

**Q: "What's top-3 accuracy?"**  
A: If we rank the model's confidence for all 6 predictions and take the top 3, 94% of the time the true label is in that top 3. It shows the model "knows" the right answer but ranks it lower.

**Q: "Should we just use top-3 accuracy instead of top-1?"**  
A: Top-1 is the real-world metric (the model must pick one). Top-3 shows model confidence; if it's high (94% vs 73%), we might improve by tuning thresholds or re-weighting classes.

---

## One-Liner Summary for Slides

> **"Transformer model achieves 73% accuracy on 6-class crisis classification. Class3 is learned well (86% recall); Class5 struggles (22% recall). Key action: collect more Class5 data and investigate Class0–Class4 overlap."**

