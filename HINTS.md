# Hints and Guidance

This document provides hints for the take-home assignment. **Try to solve challenges independently first**, then refer here if you get stuck or want to validate your thinking.

## 🎯 Getting Started

### Understanding the Problem

This is a credit risk modeling problem. Key questions to ask yourself:
- What are you predicting? (12-month default probability)
- When would this prediction be made? (At loan origination)
- What information would be available at that time?
- How will the model be used in production?

### Data Quality Hints

**Look for:**
- Unexpected values or patterns in numerical columns
- Values that seem like placeholders or error codes
- Missing data patterns (is missingness informative?)
- Distribution shifts across time (vintages)

**Questions to consider:**
- Why might some values be encoded differently than standard NaN?
- Does missing information itself carry a signal?
- How should you handle different types of missingness?

### Validation Strategy Hints

**Think about:**
- The data has a temporal structure (vintages)
- Credit risk models predict future behavior
- What would constitute a fair test of the model's ability?

**Questions to consider:**
- If you trained on all available data randomly shuffled, what information might leak?
- How do you simulate a realistic production scenario?
- How stable is performance across different time periods?
- What validation approach would give you confidence in future performance?

### Feature Selection Hints

**Critical question:** What information is actually available when making a prediction?

**Think about the timeline:**
1. Loan application is received
2. Decision is made (approve/deny)
3. Loan originates (if approved)
4. Performance is observed over time

**At what point in this timeline would you use your model?**

**Red flags to watch for:**
- Features with suspiciously high predictive power
- Features that describe events happening AFTER the prediction point
- Information that would only be known for completed loans

**Validation approach:**
- Train models with different feature sets
- Compare performance - if dropping a feature causes a dramatic drop, ask why
- Would this feature be available in a production system at decision time?

### Class Imbalance Hints

**Observe the target distribution:**
- What percentage of loans default?
- What happens if your model always predicts "no default"?

**Consider:**
- Techniques for handling imbalanced datasets
- Whether certain samples should be weighted differently
- What metrics are appropriate for imbalanced problems
- The business cost of false positives vs false negatives

**Evaluation questions:**
- Is accuracy a good metric here? Why or why not?
- What metrics better capture performance on the minority class?
- How do you choose an operating threshold?
- What are the tradeoffs between precision and recall?

---

## 🚀 Bonus Section Hints

### Metaflow Pipeline Structure

**Key concepts:**
1. **Steps**: Each step is a function decorated with `@step`
2. **Artifacts**: Data passed between steps via `self.variable`
3. **Parameters**: Run-time configuration (e.g., vintage split date)
4. **Cards**: Visual reports attached to steps

**Basic commands:**
```bash
# Run the pipeline
python credit_risk_flow.py run

# Run with custom parameters
python credit_risk_flow.py run --train_vintage_end 202212

# Show most recent run
python credit_risk_flow.py show

# List all runs
python credit_risk_flow.py list

# Access artifacts from a specific run
python credit_risk_flow.py output <run_id>
```

**Hints for implementation:**

1. **Start simple**: Get the basic pipeline working first
2. **Add logging**: Use print statements to track progress
3. **Save artifacts**: Store models, metrics, and plots
4. **Parameter sweeps**: Try multiple configurations
5. **Error handling**: Wrap steps in try/except

**Advanced features to consider:**
- `@retry` decorator for fault tolerance
- `@resources` for compute requirements
- `@batch` for parallel processing
- Custom decorators for timing/logging

---

## 💡 General Tips

### Time Management

**Suggested allocation (6-8 hours total):**
- EDA and data quality (1-2 hours)
- Feature engineering (1-2 hours)
- Model training and tuning (2-3 hours)
- Evaluation and documentation (1-2 hours)
- Code cleanup and testing (1 hour)

**If short on time, prioritize:**
1. ✅ Proper train/test split (vintage-based)
2. ✅ Handling special values correctly
3. ✅ Avoiding data leakage
4. ✅ One well-tuned model > multiple poor models
5. ✅ Clear documentation of your approach

### Code Quality

**We're looking for:**
- Modular functions (not one giant notebook cell)
- Clear variable names
- Comments explaining non-obvious decisions
- Reproducible results (set random seeds!)
- Proper error handling

**Example of good code structure:**
```python
def handle_special_values(df, config):
    """
    Replace special values with appropriate imputations.
    
    Args:
        df: Input dataframe
        config: Dict with column names and special values
    
    Returns:
        DataFrame with special values handled
    """
    df = df.copy()
    
    for col, special_val in config.items():
        if col in df.columns:
            # Create missing indicator
            df[f'{col}_missing'] = (df[col] == special_val).astype(int)
            
            # Impute with median
            median_val = df[df[col] != special_val][col].median()
            df[col] = df[col].replace(special_val, median_val)
    
    return df

# Usage
special_values_config = {
    'fico_score': 99999,
    'income': -1,
    'inquiries_last_6m': 99
}
df = handle_special_values(df, special_values_config)
```

### Things to Keep in Mind

1. ✅ Think carefully about your validation strategy
2. ✅ Consider what features should and shouldn't be used
3. ✅ Address class distribution in your approach
4. ✅ Handle data quality issues appropriately
5. ✅ Avoid overfitting - test on truly held-out data
6. ✅ Document your reasoning and assumptions
7. ✅ Show your thought process, not just final results

### What We're Really Testing

**Technical skills:**
- Understanding of credit risk modeling concepts
- Ability to handle messy, real-world data
- Proper ML methodology (train/val/test, cross-validation)
- Feature engineering creativity

**Soft skills:**
- Communication (can you explain your decisions?)
- Attention to detail (did you notice the leakage features?)
- Problem-solving (how did you handle challenges?)
- Production readiness (is your code maintainable?)

---

## ❓ FAQ

**Q: Can I use external data sources?**
A: Please use only the provided dataset.

**Q: How much feature engineering is expected?**
A: Quality over quantity. Thoughtful features with clear rationale are more valuable than many random transformations.

**Q: Should I tune hyperparameters extensively?**
A: Some tuning is good, but focus more on methodology and code quality. Document your approach.

**Q: What if I'm unsure about something?**
A: Ask! Part of the evaluation is how you handle ambiguity and seek clarification.

**Q: What if my model performance isn't great?**
A: Document what you tried and your hypotheses about what worked or didn't work. The thought process matters.

**Q: How important is the bonus section?**
A: Truly optional. A strong core submission is better than rushing through both parts.

---

Good luck! Remember, we're more interested in your thought process and approach than perfect results. 🚀

