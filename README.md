# Credit Risk Modeling - Take-Home Assignment

**Role:** Data Scientist 
**Estimated Time:** 6–8 hours (Core) + 2-4 hours (Bonus)

## 🎯 Objective

Build and document a production-ready modeling workflow that predicts the probability of 12-month loan default, using a realistic, vintage-based credit dataset.

Your solution should demonstrate:
- **Modeling skill**: Sound methodology, proper validation, and evaluation
- **Engineering maturity**: Clean, modular, reproducible code
- **Domain knowledge**: Understanding of credit risk concepts (vintages, out-of-time validation, data leakage)

---

## 📁 Project Structure

```
Credit-Risk-Modeling-Take-Home/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files to ignore in git
│
├── data/
│   └── credit_risk_data_enhanced.csv   # Dataset for modeling
│
├── notebooks/
│   └── modeling.ipynb                  # Your main deliverable
│
├── src/                                # (Optional) Modular code
│   ├── __init__.py
│   ├── preprocessing.py                # Feature engineering utilities
│   ├── modeling.py                     # Model training utilities
│   └── evaluation.py                   # Evaluation metrics
│
├── scripts/
│   └── generate_enhanced_data.py       # Data generation script -- Assuming this created the .csv
│
└── outputs/                            # (Optional) Save models/plots here
```

---

## 🧮 Part 1: Core Modeling Challenge

### Dataset: `data/credit_risk_data_enhanced.csv`

Each record represents a loan at origination. The dataset spans **multiple vintages** (2021-01 to 2024-09) with **~3% default rate**.

| Column | Type | Description | Special Values |
|--------|------|-------------|----------------|
| `loan_id` | str | Unique loan identifier | - |
| `origination_date` | date | Loan origination date | - |
| `vintage` | str | Origination month (YYYYMM) | - |
| `months_on_book` | int | Months since origination | - |
| `fico_score` | int | Credit score (550–850) | **99999 = Missing** |
| `income` | int | Annual income (USD) | **-1 = Missing** |
| `debt_to_income` | float | Ratio of debt to income | - |
| `num_open_trades` | int | Number of active credit trades | - |
| `utilization_rate` | float | Revolving credit utilization (0-1) | - |
| `inquiries_last_6m` | int | Credit inquiries in past 6 months | **99 = Missing** |
| `loan_amount` | int | Loan principal (USD) | - |
| `term` | int | Loan term in months (36 or 60) | - |
| `apr` | float | Annual percentage rate | - |
| `channel` | str | Origination channel | - |
| `product_type` | str | Loan purpose | - |
| `employment_length` | int | Years of employment | - |
| `state` | str | Borrower's state | - |
| `age` | int | Borrower's age | - |
| `days_past_due_current` | int | Current DPD status | - |
| `total_payments_to_date` | float | Payments made | - |
| `sample_weight` | float | Sample importance weight | - |
| `default_12m` | int | **Target variable** (1 = default within 12 months) | - |

### Key Challenges:

Your solution should address these common real-world modeling challenges:

1. **Data Quality**: The dataset may contain data quality issues common in production systems. Identify and handle them appropriately.

2. **Validation Strategy**: Design an appropriate train/validation/test strategy for credit risk models. Consider the temporal nature of the data.

3. **Class Imbalance**: Consider how the target distribution may affect modeling choices and evaluation.

4. **Feature Selection**: Not all features may be appropriate for a model predicting default at loan origination. Think carefully about what information would actually be available at decision time.

### Tasks:

1. **Exploratory Data Analysis**
   - Understand the data structure and distributions
   - Identify any data quality issues
   - Analyze target variable characteristics
   - Examine relationships between features and target

2. **Data Preprocessing & Feature Engineering**
   - Handle data quality issues appropriately
   - Select and engineer relevant features
   - Create encodings for categorical variables
   - Consider interactions or transformations

3. **Model Development**
   - Design an appropriate validation strategy
   - Train and tune at least two different models
   - Make informed choices about handling class distribution
   - Document your modeling decisions

4. **Model Evaluation**
   - Evaluate models using appropriate metrics
   - Analyze model performance and stability
   - Interpret feature importance
   - Assess model calibration

5. **Summary & Recommendations**
   - Summarize your approach and key findings
   - Discuss model limitations and assumptions
   - Propose next steps for production deployment
   - Highlight any concerns or areas needing clarification

### Deliverable: 
`notebooks/modeling.ipynb` (or equivalent) with all outputs saved

---

## 🚀 Part 2: BONUS - Production Pipeline (Optional)

**Estimated Time**: 2-4 hours additional

Build a **Metaflow** or **similar workflow orchestration** pipeline that:

1. **Data Loading**: Load and validate the dataset
2. **Feature Engineering**: Apply transformations as a reproducible step
3. **Model Training**: Train model with proper experiment tracking
4. **Model Evaluation**: Generate metrics and artifacts
5. **Model Versioning**: Save model with metadata

### Requirements:

- Use [Metaflow](https://metaflow.org/) or similar (Prefect, Airflow, etc.)
- Each step should be a separate, testable function
- Include proper logging and error handling
- Save model artifacts and evaluation metrics
- **Bonus points**: Add parameter sweeps or A/B testing capability

### Example Metaflow Structure:

```python
from metaflow import FlowSpec, step, Parameter

class CreditRiskFlow(FlowSpec):
    
    vintage_split = Parameter('vintage_split', default='202301')
    
    @step
    def start(self):
        """Load and validate data"""
        self.next(self.preprocess)
    
    @step
    def preprocess(self):
        """Feature engineering and handling special values"""
        self.next(self.train)
    
    @step
    def train(self):
        """Train model with out-of-time validation"""
        self.next(self.evaluate)
    
    @step
    def evaluate(self):
        """Generate metrics and save artifacts"""
        self.next(self.end)
    
    @step
    def end(self):
        """Pipeline complete"""
        pass

if __name__ == '__main__':
    CreditRiskFlow()
```

**Deliverable**: 
- `src/credit_risk_flow.py` or similar
- Documentation on how to run the pipeline
- Example of viewing results from different runs

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <repository-url>
cd Credit-Risk-Modeling-Take-Home
```

### 2. Set up Python environment
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate the dataset (already done)
```bash
# The dataset is already generated, but you can regenerate it:
cd scripts
python generate_enhanced_data.py
```

### 4. Start working
```bash
# Launch Jupyter
jupyter notebook notebooks/modeling.ipynb

# Or use JupyterLab
jupyter lab
```

---

## 📊 Evaluation Criteria

Your submission will be evaluated on:

- **Technical Approach**: Sound modeling methodology and appropriate techniques
- **Data Understanding**: Awareness of data quality issues and proper handling
- **Feature Engineering**: Thoughtful feature creation and selection
- **Model Performance**: Model quality and evaluation rigor
- **Code Quality**: Clean, readable, and reproducible code
- **Communication**: Clear documentation, insights, and justification of decisions
- **Production Readiness**: Considerations for real-world deployment

### Bonus Section (Optional):
- **Engineering Best Practices**: Well-structured, testable, production-ready code
- **Workflow Management**: Proper orchestration and experiment tracking

---

## 📤 Submission Instructions

1. Complete your analysis in `notebooks/modeling.ipynb`
2. (Optional) Add modular code to `src/` 
3. (Optional) Add Metaflow pipeline for bonus section
4. Push your code to a GitHub repository (or provide a zip file)
5. Ensure all notebook outputs are saved so we can review without running
6. Include a summary section at the end with:
   - Key findings
   - Model performance summary
   - Identified data quality issues
   - Recommendations for improvement
   - (If bonus) Instructions to run your pipeline

---

## ⚡ Tips

- **Start simple**: Get a baseline working before adding complexity
- **Document your thinking**: Explain the rationale behind your decisions
- **Show your work**: Include visualizations and intermediate results
- **Ask questions**: If anything is unclear, please reach out
- **Time management**: Focus on demonstrating your core skills first

---

## ❓ Questions?

If you have any questions about the assignment, please reach out to elly.arabmakki@bestegg.com

Good luck! 🎉
