"""
Dataset for Global Insurance Brokers Performance Evaluation (2024)
Extracted directly from Applied Soft Computing Journal 190 (2026) 114557.
"""

import numpy as np

# Table 4: Criteria Information (Code, Name, Category, Direction: 'max' for benefit, 'min' for cost)
CRITERIA_METADATA = [
    {"code": "C1", "name": "Current Ratio", "category": "Ratios", "type": "max", "definition": "Current Assets / Current Liabilities"},
    {"code": "C2", "name": "Quick Ratio", "category": "Ratios", "type": "max", "definition": "(Current Assets - Inventory) / Current Liabilities"},
    {"code": "C3", "name": "Cash Ratio", "category": "Ratios", "type": "max", "definition": "(Cash + Cash Equivalents) / Current Liabilities"},
    {"code": "C4", "name": "Debt/Equity", "category": "Ratios", "type": "min", "definition": "Total Debt / Total Equity"},
    {"code": "C5", "name": "Debt to Capital", "category": "Ratios", "type": "min", "definition": "Total Debt / (Total Debt + Total Equity)"},
    {"code": "C6", "name": "Financial Leverage (Assets/Equity)", "category": "Ratios", "type": "min", "definition": "Total Assets / Total Equity"},
    {"code": "C7", "name": "ROA (%)", "category": "Ratios", "type": "max", "definition": "(Net Income / Total Assets) * 100"},
    {"code": "C8", "name": "ROE (%)", "category": "Ratios", "type": "max", "definition": "(Net Income / Shareholders' Equity) * 100"},
    {"code": "C9", "name": "Operating Margin (%)", "category": "Ratios", "type": "max", "definition": "(Operating Income / Revenue) * 100"},
    {"code": "C10", "name": "Net Profit Margin (%)", "category": "Ratios", "type": "max", "definition": "(Net Income / Revenue) * 100"},
    {"code": "C11", "name": "EBITDA Margin (%)", "category": "Ratios", "type": "max", "definition": "(EBITDA / Revenue) * 100"},
    {"code": "C12", "name": "Interest Coverage", "category": "Ratios", "type": "max", "definition": "EBIT / Interest Expense"},
    {"code": "C13", "name": "Asset Turnover", "category": "Ratios", "type": "max", "definition": "Revenue / Total Assets"},
    {"code": "C14", "name": "Operating Cash Flow / Net Income", "category": "Ratios", "type": "max", "definition": "Operating Cash Flow / Net Income"},
    {"code": "C15", "name": "Current Assets / Total Assets", "category": "Ratios", "type": "max", "definition": "Current Assets / Total Assets"},
    {"code": "C16", "name": "Operating Cash Flow", "category": "Cash Flow", "type": "max", "definition": "Net Cash from Operating Activities"},
    {"code": "C17", "name": "Investing Cash Flow", "category": "Cash Flow", "type": "min", "definition": "Net Cash from Investing Activities"},
    {"code": "C18", "name": "Financing Cash Flow", "category": "Cash Flow", "type": "min", "definition": "Net Cash from Financing Activities"},
    {"code": "C19", "name": "Total Assets", "category": "Financial Statement Items", "type": "max", "definition": "Sum of Current and Non-Current Assets"},
    {"code": "C20", "name": "Revenue", "category": "Financial Statement Items", "type": "max", "definition": "Total Sales or Service Income"},
    {"code": "C21", "name": "Income from Operations", "category": "Financial Statement Items", "type": "max", "definition": "Revenue - Operating Expenses"},
    {"code": "C22", "name": "Earnings Per Share", "category": "Financial Statement Items", "type": "max", "definition": "(Net Income - Preferred Dividends) / W.A. Shares"},
    {"code": "C23", "name": "Net Income", "category": "Financial Statement Items", "type": "max", "definition": "Total Earnings After Expenses, Interest, and Taxes"},
    {"code": "C24", "name": "Dividends Paid", "category": "Financial Statement Items", "type": "max", "definition": "Cash Distributed to Shareholders"},
    {"code": "C25", "name": "Interest Expense", "category": "Financial Statement Items", "type": "min", "definition": "Cost of Borrowing"},
    {"code": "C26", "name": "Mean of Return", "category": "Stock Market", "type": "max", "definition": "Average Stock Return Over a Period"},
    {"code": "C27", "name": "Std of Return", "category": "Stock Market", "type": "min", "definition": "Volatility of Stock Returns"},
    {"code": "C28", "name": "Yearly Beta", "category": "Stock Market", "type": "min", "definition": "Volatility Relative to Market (Beta=1)"},
    {"code": "C29", "name": "Credit Rating", "category": "Financial Notes", "type": "min", "definition": "Creditworthiness (Lower value is better: 1 is AAA+)"},
    {"code": "C30", "name": "Number of Employees", "category": "Operational", "type": "max", "definition": "Total Headcount of Company Workforce"}
]

# Table 5: Expert Preferences organized into Significance Levels Q1 to Q4
# Each entry is (Criterion Code, (lower, modal, upper) Triangular Fuzzy Rating)
EXPERT_PREFERENCES_TABLE5 = {
    "Q1": [
        ("C29", (0.00, 0.90, 2.00)),
        ("C4",  (1.00, 2.00, 3.00)),
        ("C6",  (2.00, 3.20, 5.00)),
        ("C5",  (4.00, 4.60, 6.00)),
        ("C12", (5.00, 5.70, 7.50)),
        ("C1",  (6.00, 6.70, 8.50)),
        ("C3",  (7.50, 8.20, 10.00)),
        ("C2",  (8.00, 10.00, 12.00)),
        ("C15", (9.00, 10.80, 13.00))
    ],
    "Q2": [
        ("C8",  (0.00, 0.90, 2.00)),
        ("C7",  (1.00, 1.70, 3.00)),
        ("C10", (2.00, 2.80, 4.50)),
        ("C9",  (3.00, 4.10, 5.50)),
        ("C11", (4.50, 5.20, 6.00)),
        ("C22", (6.00, 6.60, 8.00)),
        ("C26", (7.00, 8.20, 9.50))
    ],
    "Q3": [
        ("C14", (0.00, 0.90, 2.00)),
        ("C16", (1.00, 2.40, 4.00)),
        ("C13", (2.00, 3.50, 5.00)),
        ("C17", (4.00, 5.80, 7.00)),
        ("C18", (6.00, 7.20, 8.00))
    ],
    "Q4": [
        ("C28", (0.00, 0.90, 2.00)),
        ("C27", (1.00, 2.00, 3.50)),
        ("C19", (2.00, 3.10, 5.00)),
        ("C20", (3.00, 4.30, 5.50)),
        ("C23", (5.00, 5.60, 7.00)),
        ("C21", (5.50, 6.60, 8.00)),
        ("C25", (7.50, 8.40, 10.00)),
        ("C24", (8.00, 9.60, 12.00)),
        ("C30", (11.00, 12.20, 14.00))
    ]
}

# Table 7: Baseline Fuzzy and Crisp Weights derived from F-LBWA (Elasticity theta = 14.01)
BASELINE_WEIGHTS_TABLE7 = {
    "C1":  {"fuzzy": (0.045, 0.051, 0.056), "crisp": 0.0509},
    "C2":  {"fuzzy": (0.039, 0.044, 0.051), "crisp": 0.0444},
    "C3":  {"fuzzy": (0.043, 0.048, 0.052), "crisp": 0.0475},
    "C4":  {"fuzzy": (0.060, 0.066, 0.074), "crisp": 0.0664},
    "C5":  {"fuzzy": (0.051, 0.057, 0.062), "crisp": 0.0567},
    "C6":  {"fuzzy": (0.054, 0.061, 0.069), "crisp": 0.0615},
    "C7":  {"fuzzy": (0.033, 0.036, 0.038), "crisp": 0.0356},
    "C8":  {"fuzzy": (0.034, 0.037, 0.040), "crisp": 0.0367},
    "C9":  {"fuzzy": (0.030, 0.033, 0.036), "crisp": 0.0330},
    "C10": {"fuzzy": (0.031, 0.034, 0.037), "crisp": 0.0343},
    "C11": {"fuzzy": (0.030, 0.032, 0.034), "crisp": 0.0319},
    "C12": {"fuzzy": (0.047, 0.054, 0.059), "crisp": 0.0535},
    "C13": {"fuzzy": (0.022, 0.023, 0.025), "crisp": 0.0233},
    "C14": {"fuzzy": (0.023, 0.025, 0.026), "crisp": 0.0247},
    "C15": {"fuzzy": (0.038, 0.043, 0.048), "crisp": 0.0428},
    "C16": {"fuzzy": (0.022, 0.024, 0.026), "crisp": 0.0239},
    "C17": {"fuzzy": (0.021, 0.022, 0.024), "crisp": 0.0222},
    "C18": {"fuzzy": (0.020, 0.021, 0.023), "crisp": 0.0216},
    "C19": {"fuzzy": (0.017, 0.018, 0.019), "crisp": 0.0179},
    "C20": {"fuzzy": (0.017, 0.018, 0.019), "crisp": 0.0176},
    "C21": {"fuzzy": (0.016, 0.017, 0.018), "crisp": 0.0169},
    "C22": {"fuzzy": (0.028, 0.031, 0.033), "crisp": 0.0306},
    "C23": {"fuzzy": (0.016, 0.017, 0.018), "crisp": 0.0172},
    "C24": {"fuzzy": (0.015, 0.016, 0.017), "crisp": 0.0161},
    "C25": {"fuzzy": (0.015, 0.016, 0.018), "crisp": 0.0164},
    "C26": {"fuzzy": (0.027, 0.029, 0.032), "crisp": 0.0293},
    "C27": {"fuzzy": (0.017, 0.018, 0.019), "crisp": 0.0183},
    "C28": {"fuzzy": (0.018, 0.019, 0.020), "crisp": 0.0186},
    "C29": {"fuzzy": (0.073, 0.076, 0.079), "crisp": 0.0757},
    "C30": {"fuzzy": (0.015, 0.016, 0.017), "crisp": 0.0155}
}

# 5 Global Insurance Brokerage Firms (Listed in S&P 500)
INSURANCE_BROKERS = [
    {"code": "AJG", "name": "Arthur J. Gallagher & Co."},
    {"code": "AON", "name": "Aon plc"},
    {"code": "BRO", "name": "Brown & Brown, Inc."},
    {"code": "MMC", "name": "Marsh & McLennan Companies, Inc."},
    {"code": "WTW", "name": "Willis Towers Watson Public Limited Company"}
]

# Table 8: Raw Decision Matrix for Year 2024
# Shape: (5 alternatives, 30 criteria)
DECISION_MATRIX_2024 = np.array([
    #  AJG          AON          BRO          MMC          WTW
    [1.51,        1.02,        1.10,        1.13,        1.20],       # C1: Current Ratio
    [0.65,        0.22,        0.25,        0.49,        0.35],       # C2: Quick Ratio
    [0.51,        0.05,        0.11,        0.12,        0.15],       # C3: Cash Ratio
    [0.65,        2.70,        0.63,        1.47,        0.66],       # C4: Debt/Equity
    [0.39,        0.73,        0.39,        0.60,        0.40],       # C5: Debt to Capital
    [3.18,        7.77,        2.74,        4.17,        3.45],       # C6: Financial Leverage
    [2.28,        5.55,        5.64,        7.19,       -0.35],       # C7: ROA (%)
    [7.25,       43.14,       15.43,       30.00,       -1.22],       # C8: ROE (%)
    [19.53,      24.43,       23.10,       23.78,        6.31],       # C9: Operating Margin (%)
    [12.66,      17.33,       20.67,       16.60,       -0.99],       # C10: Net Profit Margin (%)
    [26.81,      28.80,       35.57,       26.83,       10.91],       # C11: EBITDA Margin (%)
    [5.92,        4.87,        7.70,        8.31,        2.38],       # C12: Interest Coverage
    [0.18,        0.32,        0.27,        0.43,        0.36],       # C13: Asset Turnover
    [1.77,        1.12,        1.18,        1.06,      -15.43],       # C14: OCF / Net Income
    [68.70,      47.90,       39.30,       39.20,       54.60],       # C15: Current Assets / Total Assets
    [2582.90,   3035.00,     1174.00,     4302.00,     1512.00],      # C16: Operating Cash Flow
    [-1587.40, -2833.00,     -898.00,    -8821.00,      250.00],      # C17: Investing Cash Flow
    [13052.70,   796.00,      -64.00,     4455.00,     -459.00],      # C18: Financing Cash Flow
    [64255.20, 48965.00,    17612.00,    56481.00,    27681.00],      # C19: Total Assets
    [11554.90, 15698.00,     4805.00,    24458.00,     9930.00],      # C20: Revenue
    [2256.70,   3835.00,     1110.00,     5816.10,      626.60],      # C21: Operating Income
    [6.63,       12.55,        3.48,        8.26,       -0.96],       # C22: EPS
    [1462.90,   2720.50,      993.20,     4060.00,      -98.30],      # C23: Net Income
    [525.40,     562.00,      154.00,     1513.00,      354.00],      # C24: Dividends Paid
    [381.20,     787.50,      144.20,      699.90,      263.30],      # C25: Interest Expense
    [0.098571,  0.091230,    0.149603,    0.049325,    0.110040],    # C26: Mean Return
    [0.010799,  0.012233,    0.011367,    0.008720,    0.011081],    # C27: Std Return
    [0.73,        0.85,        0.65,        0.40,        0.66],       # C28: Beta
    [4.00,        4.00,        4.00,        3.00,        4.00],       # C29: Credit Rating
    [56000.0,   60000.0,     17403.0,     90000.0,     48900.0]       # C30: Number of Employees
]).T  # Transpose to shape (5 alternatives, 30 criteria)


def get_dataset():
    """
    Returns standard evaluation inputs:
    - X: raw decision matrix (shape: 5 alternatives, 30 criteria)
    - criteria_types: list of 'max' or 'min' for each criterion
    - weights: numpy array of crisp weights (sum = 1.0)
    - alternative_names: list of broker codes ['AJG', 'AON', 'BRO', 'MMC', 'WTW']
    - criteria_codes: list of criteria codes ['C1', ..., 'C30']
    """
    X = DECISION_MATRIX_2024.copy()
    criteria_types = [c["type"] for c in CRITERIA_METADATA]
    criteria_codes = [c["code"] for c in CRITERIA_METADATA]
    alternative_names = [b["code"] for b in INSURANCE_BROKERS]
    
    weights = np.array([BASELINE_WEIGHTS_TABLE7[code]["crisp"] for code in criteria_codes])
    weights = weights / np.sum(weights)  # strict simplex normalization
    
    return {
        "X": X,
        "criteria_types": criteria_types,
        "criteria_codes": criteria_codes,
        "weights": weights,
        "alternative_names": alternative_names,
        "metadata": CRITERIA_METADATA
    }
