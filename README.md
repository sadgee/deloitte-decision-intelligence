<div align="center">

<br/>

# 🟢 Deloitte Decision Intelligence Platform

### *Turning Data into Trusted Decisions*
#### A Hybrid ML + AI System for Investment Decision-Making

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![AWS](https://img.shields.io/badge/AWS-Bedrock%20·%20EC2%20·%20S3-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%204.5-D97706?style=for-the-badge&logoColor=white)](https://anthropic.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-E65100?style=for-the-badge&logoColor=white)](https://xgboost.readthedocs.io)

<br/>

> **Capstone Project 2026 · Tepper School of Business, Carnegie Mellon University**
>
> Kyle Chen &nbsp;·&nbsp; Sadgee Pandey &nbsp;·&nbsp; Anastasia Raicevic &nbsp;·&nbsp; Ruo Yu Wang
>
> *Sponsored by Deloitte*

<br/>

```
⚡  ROI prediction in under 10 seconds
🎯  75.2% directional accuracy across 120,000 decisions
🧠  ML computes · GenAI interprets · Humans decide
```

<br/>

</div>

---

## 📌 The Problem

> *"Even in organizations with advanced analytics, executives hesitate to act on model-driven insights — because they can't understand why the model said what it said."*

Traditional ML models produce **numbers**. They don't produce **decisions**.

A predicted ROI of +14% means nothing to an executive if they can't see what's driving it, whether the macro environment supports it, or what could go wrong. This gap between analytical output and actionable decision is what this platform is built to close.

<br/>

---

## 💡 The Solution

The **Deloitte Decision Intelligence Platform** combines three layers into one seamless workflow:

<br/>

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│   🔢  PREDICT       │────▶│   🧠  EXPLAIN       │────▶│   📊  GROUND        │
│                     │     │                     │     │                     │
│  GBM + XGBoost      │     │  Claude Sonnet 4.5  │     │  Industry           │
│  + LightGBM         │     │  via AWS Bedrock    │     │  Benchmarks         │
│                     │     │                     │     │                     │
│  ROI · Risk ·       │     │  Market signals ·   │     │  Avg · Top Q ·      │
│  Confidence         │     │  Executive report   │     │  Bottom Q           │
│                     │     │                     │     │                     │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

<br/>

**Output:** A complete decision package — predicted ROI, confidence score, risk level, investment verdict, signal breakdown, and a Claude-generated executive narrative — delivered in under 10 seconds.

<br/>

---

## 🖥️ Demo Output

*Retail Pricing decision evaluated under adverse macro conditions:*

```
╔══════════════════════════════════════════════════════════════════════╗
║  PREDICTED ROI     AI CONFIDENCE     RISK LEVEL     VERDICT         ║
║    −11.2%               56%             HIGH        DO NOT PROCEED  ║
╚══════════════════════════════════════════════════════════════════════╝

  AI SIGNALS  (extracted from market news by Claude)
  ├─ Consumer Sentiment    −0.70   ████████░░  Bearish outlook
  ├─ Competitive Pressure   0.80   ████████░░  High competition
  └─ External Risk         FLAG    ██████████  Rate hike risk identified

  ROI BREAKDOWN
  ├─ Base ML model estimate                    + 8.0%
  ├─ Sentiment adjustment   (−0.70 × 0.18)   −12.6%
  ├─ External risk penalty  ( 1    × 0.12)   −12.0%
  ├─ Competition adjustment ( 0.30 × 0.06)   − 1.8%
  └─ Final predicted ROI                     −11.2%
```

<br/>

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                        REACT FRONTEND                                ║
║                   AWS EC2  ·  Port 3000                             ║
╚═════════════════════════════╦════════════════════════════════════════╝
                              ║  POST /predict
╔═════════════════════════════╩════════════════════════════════════════╗
║                      FASTAPI BACKEND  (EC2 · Port 8000)              ║
║                                                                      ║
║   Step 1          Step 2              Step 3            Step 4       ║
║  ┌──────────┐   ┌────────────────┐  ┌────────────┐  ┌───────────┐  ║
║  │ Validate │──▶│ Extract Signals│─▶│ ML Predict │─▶│ Apply AI  │  ║
║  │  Inputs  │   │  via Claude AI │  │  Ensemble  │  │  Signals  │  ║
║  └──────────┘   └────────────────┘  └────────────┘  └─────┬─────┘  ║
║                                                            │         ║
║   Step 5                    Step 6                Step 7  │         ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌───────────▼──────┐  ║
║  │ Confidence Score │  │ Build Transparncy│  │ Generate Report  │  ║
║  │  Calculation     │  │ ROI Breakdown +  │  │  via Claude AI   │  ║
║  │                  │  │ Key Factors      │  │                  │  ║
║  └──────────────────┘  └──────────────────┘  └──────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════╝
         │                        │                       │
   ┌─────▼──────┐          ┌──────▼───────┐       ┌──────▼──────┐
   │   AWS S3   │          │ AWS Bedrock  │       │  SageMaker  │
   │  roi_model │          │   Claude     │       │  (future)   │
   │   .pkl     │          │  Sonnet 4.5  │       │  retraining │
   └────────────┘          └──────────────┘       └─────────────┘
```

<br/>

---

## ⚙️ Backend — Deep Dive

> `backend/main.py` — FastAPI + uvicorn · AWS EC2 Port 8000

The backend is the core of the platform. On every `/predict` request it runs a 7-step pipeline that goes from raw user input to a complete, explainable decision package.

<br/>

### 🔁 Request Pipeline

**Step 1 — Input Validation**

All incoming values are clamped to safe ranges before entering the model. VIX is capped, margins are clipped to `[−1, 1]`, and revenue and investment values are floored at zero. This prevents out-of-distribution inputs from producing nonsensical outputs.

<br/>

**Step 2 — AI Signal Extraction via Claude**

The `news_text` field is sent to Claude Sonnet 4.5 via AWS Bedrock. Claude reads the news paragraph as a whole and returns exactly one JSON object:

```json
{
  "sentiment":      -0.70,   // −1.0 (very negative)  →  +1.0 (very positive)
  "competition":     0.80,   //  0.0 (low)             →   1.0 (high)
  "external_risk":   1       //  0 = no event          →   1 = flagged
}
```

This is not rule-based extraction. There is no keyword list or sentiment dictionary. Claude makes a judgment the same way an analyst reading a Bloomberg article would — holistically. A regex fallback handles cases where Claude wraps the output in markdown code fences. If no news is provided, signals default to neutral `(0, 0.5, 0)` and the confidence score is penalized by 8 percentage points.

<br/>

**Step 3 — Feature Engineering**

Four derived features are computed at inference time and must exactly match those engineered during training:

| Feature | Formula | Why it matters |
|---|---|---|
| `investment_ratio` | `Investment ÷ Revenue` | Normalizes investment size to company scale |
| `macro_stress` | `VIX×0.4 + Unemployment×0.3 + (−GDP)×0.3` | Compresses macro environment into a single risk signal |
| `vol_vix` | `Vol_30D × VIX` | Captures compounding market instability |
| `margin_x_investment` | `Margin × Investment` | Reflects financial capacity to absorb the investment cost |

<br/>

**Step 4 — ML Ensemble Inference**

The 22-feature input vector is passed to the voting ensemble. The raw probability output is scaled by a compression factor of `0.06` to bring it into a realistic range of approximately −50% to +80% — necessary because synthetic training data inflated raw outputs significantly.

```
Base ROI = clip(raw_output × 0.06,  −0.50,  +0.80)
```

<br/>

**Step 5 — AI Signal Adjustment**

The three Claude-extracted signals are applied as explicit post-prediction adjustments on top of the base ML estimate. This formula is shown verbatim in the frontend transparency panel:

```
Final ROI = Base ROI
          + (Sentiment     × 0.18)         ← strongest weight: direct demand signal
          − (External Risk × 0.12)         ← binary penalty for categorical risk
          − ((Competition  − 0.5) × 0.06)  ← centered at neutral; ±deviation shifts result
```

<br/>

**Step 6 — Confidence Score**

A multi-factor composite score built from four weighted components:

```
Market Stability  (VIX)    35%   VIX < 20 → full  ·  20–30 → partial  ·  > 30 → low
GDP Direction              30%   Positive → full  ·  Negative → reduced by magnitude
Signal Strength            20%   |sentiment| + |competition − 0.5|
Base Prior                 15%   Aggregate ensemble accuracy on held-out test set

Penalties:
  External risk flagged    −10 percentage points
  No news text provided     −8 percentage points
```

<br/>

**Step 7 — Executive Report Generation**

A second Bedrock call sends the complete decision context — all structured inputs, AI signals, model outputs, and confidence score — to Claude. Claude returns a three-section report written for a senior executive audience:

- **Core Risks** — the specific combination of inputs creating downside exposure
- **Recommendations** — concrete, input-specific guidance (not generic advice)
- **Optimal Timing** — macro triggers that would make this decision worth revisiting

The report is regenerated on every request and is specific to the exact inputs provided.

<br/>

### 🔑 Key Design Decisions

> **Claude never touches the numbers.**

Every value in the response comes from the ML ensemble or from deterministic formulas. Claude only reads text and interprets outputs the model has already calculated. This eliminates hallucinated numeric outputs and makes the entire prediction path fully auditable.

> **AI signals are post-prediction adjustments — not model features.**

Including sentiment, competition, and risk as training features would have required generating synthetic news text for 120,000 training records. More importantly, keeping them outside the model makes their contribution fully visible — users see the exact numerical impact of each signal, which would be impossible if they were absorbed into the ensemble's internal weights.

> **Transparency is first-class.**

The backend doesn't just return a number — it returns a complete explanation of every factor that contributed to it. The ROI breakdown, key factors panel, decision logic, and industry benchmark are all constructed server-side so the frontend can render a full transparency panel with no additional computation.

<br/>

### 📦 Startup & Model Loading

On server startup, the backend loads the trained ML ensemble and label encoders from AWS S3 into memory:

```python
obj = s3.get_object(Bucket="deloitteml", Key="models/roi_model.pkl")
MODEL_ARTIFACTS = joblib.load(io.BytesIO(obj["Body"].read()))
```

The model is cached globally — not reloaded on every request. All AWS access is handled through the EC2 IAM role `deloitte-ec2-role`. No credentials are hardcoded anywhere.

<br/>

---

## 🤖 ML Model

### Ensemble — Three Models, One Vote

| Model | Strength |
|---|---|
| **GBM** (scikit-learn) | Strong tabular baseline; corrects residuals sequentially |
| **XGBoost** | L1/L2 regularization; handles missing values natively |
| **LightGBM** | Leaf-wise tree growth; fastest on large datasets |

Soft voting averages predicted probabilities rather than taking a majority class vote — producing more calibrated outputs when models disagree.

### Performance

```
Directional Accuracy    75.2%   ████████████████████████░░░░░░░░
Training Records        120,000 synthetic business decisions
Train / Test Split      80 / 20
Industries              Manufacturing · SaaS · Retail · Healthcare · Finance
Decision Types          Marketing · Pricing · Expansion · R&D Investment · Hiring
```

> ⚠️ The 75.2% figure reflects performance on a held-out split of the **same synthetic dataset** — not on real-world decisions. Treat as a proof-of-concept benchmark until retrained on real Deloitte client data.

<br/>

<details>
<summary><b>📋 Full Feature Set — 22 Variables (click to expand)</b></summary>

<br/>

**Company Profile**
| Feature | Type | Description |
|---|---|---|
| `Industry` | Categorical | Manufacturing, SaaS, Retail, Healthcare, Finance |
| `Company_Size` | Categorical | Small, Medium, Large |
| `Operating_Margin` | Float | e.g. `0.08` = 8% margin |
| `Baseline_Revenue` | Float | Quarterly revenue in dollars |

**Decision Parameters**
| Feature | Type | Description |
|---|---|---|
| `Decision_Type` | Categorical | Marketing, Pricing, Expansion, R&D Investment, Hiring |
| `Investment_Cost` | Float | Total investment in dollars |
| `Time_Horizon_Months` | Int | Payback / evaluation period |
| `Campaign_Intensity` | Categorical | Low, Medium, High |

**Macroeconomic Indicators**
| Feature | Type | Description |
|---|---|---|
| `GDP_Growth` | Float | Quarterly GDP growth rate |
| `Inflation_Expect` | Float | Expected inflation rate |
| `Unemployment` | Float | Current unemployment rate |
| `Interest_Rate` | Float | Prevailing interest rate |
| `VIX` | Float | CBOE Volatility Index |
| `Vol_30D` | Float | 30-day realized market volatility |
| `Mkt_Ret` | Float | Recent market return |

**Engineered Features**
| Feature | Formula |
|---|---|
| `investment_ratio` | `Investment ÷ Revenue` |
| `macro_stress` | `VIX×0.4 + Unemployment×0.3 + (−GDP)×0.3` |
| `vol_vix` | `Vol_30D × VIX` |
| `margin_x_investment` | `Operating_Margin × Investment_Cost` |

</details>

<br/>

---

## 🗂️ Project Structure

```
deloitte-decision-intelligence/
│
├── 📄  README.md
│
├── 🖥️   frontend/
│   ├── package.json
│   └── src/
│       └── App.js                  # Full React UI — Deloitte-branded, single file
│
├── ⚙️   backend/
│   ├── main.py                     # FastAPI · ML inference · Bedrock calls
│   └── requirements.txt
│
└── 📚  docs/
    ├── Team_29_Report.pdf           # Full technical report (25 pages)
    ├── Team29_Capstone_Poster.pdf   # Conference-style research poster
    └── Team29_Capstone_Final_Prez_Slides.pptx
```

<br/>

---

## 🚀 Setup & Deployment

### Prerequisites

- Python 3.9+
- Node.js 18+
- AWS account with **Bedrock enabled** in `us-east-1`
- EC2 instance with IAM role `deloitte-ec2-role` (S3 + Bedrock permissions)
- Trained model uploaded to `s3://deloitteml/models/roi_model.pkl`

<br/>

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

**EC2 background deployment:**
```bash
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
```

### Frontend

```bash
cd frontend
npm install
npm start
```

> Update `API_URL` in `App.js` to your EC2 instance IP before deploying.

### 🔐 Security

All AWS access is via the EC2 IAM role — no credentials are hardcoded.
**Never commit `.env` files, API keys, or `.pkl` model artifacts to this repository.**

<br/>

---

## 📡 API Reference

### `GET /`
```json
{ "status": "ok", "message": "Deloitte ROI Decision System Running" }
```

### `POST /predict`

<details>
<summary><b>📥 Request Body</b></summary>

<br/>

```json
{
  "Industry":            "Retail",
  "Company_Size":        "Large",
  "Operating_Margin":    0.08,
  "Baseline_Revenue":    200000000,
  "Decision_Type":       "Pricing",
  "Investment_Cost":     2000000,
  "Time_Horizon_Months": 6,
  "Campaign_Intensity":  "High",
  "GDP_Growth":         -0.02,
  "Inflation_Expect":    0.04,
  "Unemployment":        0.065,
  "Interest_Rate":       0.05,
  "VIX":                 35,
  "Vol_30D":             0.04,
  "Mkt_Ret":            -0.01,
  "External_Risk_Flag":  1,
  "news_text":           "Consumer confidence has declined for the third consecutive quarter..."
}
```
</details>

<details>
<summary><b>📤 Response Fields</b></summary>

<br/>

| Field | Type | Description |
|---|---|---|
| `roi` | `float` | Final predicted ROI after ML + AI adjustments |
| `confidence` | `float` | AI confidence score 0–1 |
| `risk_level` | `string` | `Low` · `Medium` · `Medium-High` · `High` |
| `recommendation` | `string` | `Proceed` · `Proceed with Caution` · `Do Not Proceed` |
| `consumer_sentiment` | `float` | Claude-extracted sentiment −1 to +1 |
| `competitive_pressure` | `float` | Claude-extracted competition 0–1 |
| `external_risk` | `int` | Claude-extracted risk flag 0 or 1 |
| `ai_report` | `string` | Three-section executive intelligence report |
| `roi_breakdown` | `dict` | Base ROI + each signal's contribution |
| `key_factors` | `list` | Inputs that crossed significance thresholds |
| `benchmark` | `dict` | Industry avg · top quartile · bottom quartile |

</details>

<br/>

---

## ⚠️ Limitations

| | Limitation | Detail |
|:---:|---|---|
| 🔬 | **Synthetic training data** | 75.2% is on synthetic data — not validated against real client outcomes |
| 🏭 | **5 industries only** | Out-of-distribution inputs produce less reliable outputs |
| 📏 | **Output scaling** | Compression factor `0.06` was chosen empirically; may be miscalibrated |
| 🤖 | **LLM extraction** | Claude's signal extraction cannot be audited like a deterministic algorithm |
| 👤 | **Human review required** | Every output needs human review before any real capital allocation |

<br/>

---

## 🗺️ Next Steps

```
01  Validate on real data         Retrain on actual Deloitte client engagement history
                                  — gating condition for any client-facing deployment

02  A/B test the GenAI layer      Measure whether narrative explanations measurably
                                  reduce executive forecast overrides driven by distrust

03  Industry-level accuracy       Break the 75.2% aggregate down by industry
                                  — confirm no sector is pulling the average down

04  Internal data integration     Add CRM, inventory, and override history
                                  — transforms generic tool into a client-specific asset

05  Harden infrastructure         Containerize, load balance, version models,
                                  and add rollback mechanisms for production scale
```

<br/>

---

## 📜 License & IP

This platform was developed as part of an academic capstone engagement between **Carnegie Mellon University** and **Deloitte**. Before any commercial deployment, IP ownership must be clarified in accordance with the EPA and NDA signed at the start of the engagement.

<br/>


<div align="center">

<br/>


<br/>

</div>
