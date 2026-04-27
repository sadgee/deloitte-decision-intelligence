Deloitte Decision Intelligence Platform
Turning Data into Trusted Decisions — A Hybrid ML + AI System for Investment Decision-Making

Capstone Project 2026 · Tepper School of Business, Carnegie Mellon University
Team 4: Kyle Chen · Sadgee Pandey · Anastasia Raicevic · Ruo Yu Wang
Client: Deloitte


Overview
Executives are increasingly required to make high-stakes investment decisions under uncertainty. While predictive models provide valuable forecasts, their outputs are frequently difficult to interpret and lack transparency — making it challenging for decision-makers to understand the assumptions and risks embedded in the results.
This platform addresses that gap. It combines a voting ensemble ML model with a Claude AI reasoning layer to deliver not just a prediction, but a complete decision package: predicted ROI, risk assessment, AI signal extraction from market news, and an executive intelligence report — all in under 10 seconds.
The system achieves 75.2% directional accuracy on a held-out test set of 120,000 synthetic business decisions across 5 industries and 5 decision types.

Demo
Below is a sample output for a Retail Pricing decision under adverse macro conditions:
OutputValuePredicted ROI-11.2%AI Confidence56%Risk LevelHighVerdictDo Not Proceed
AI Signals extracted from market news:

Consumer Sentiment: -0.70 (Bearish)
Competitive Pressure: 0.80 (High)
External Risk: Flagged


Architecture
┌─────────────────────────────────────────────────────────────┐
│                        React Frontend                        │
│              (EC2 Port 3000 · Deloitte UI)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ POST /predict
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend                           │
│                  (EC2 Port 8000)                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Input       │  │  AI Signal   │  │  ML Prediction   │  │
│  │  Validation  │→ │  Extraction  │→ │  Layer           │  │
│  └──────────────┘  │  (Claude via │  │  GBM + XGBoost   │  │
│                    │   Bedrock)   │  │  + LightGBM      │  │
│                    └──────────────┘  └──────────────────┘  │
│                                               │              │
│                    ┌──────────────────────────▼───────────┐ │
│                    │  Executive Report Generation          │ │
│                    │  (Claude via Bedrock)                 │ │
│                    └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼──────────────────┐
        ▼                ▼                  ▼
    AWS S3           AWS Bedrock       AWS SageMaker
  (Model Store)    (Claude Sonnet)    (Future Retraining)
How It Works — Step by Step

Consultant inputs company profile, decision parameters, and macroeconomic indicators via the React UI
Claude AI reads pasted market news text and extracts 3 quantitative signals: consumer sentiment (−1 to +1), competitive pressure (0 to 1), and external risk flag (0 or 1)
Ensemble ML model produces a base ROI estimate from 22 structured features
AI signals are applied as explicit post-prediction adjustments using the formula:

Final ROI = Base ROI + (Sentiment × 0.18) − (External Risk × 0.12) − ((Competition − 0.5) × 0.06)

Claude generates a three-section executive intelligence report: Core Risks, Recommendations, and Optimal Timing
The frontend displays the full transparency panel: ROI breakdown by signal, key factor explanations, and industry benchmark comparison


Tech Stack
LayerTechnologyFrontendReact 18, served on AWS EC2 port 3000BackendFastAPI + uvicorn, AWS EC2 port 8000ML ModelGBM + XGBoost + LightGBM voting ensemble (scikit-learn)AI LayerClaude Sonnet 4.5 via AWS BedrockModel StorageAWS S3 (serialized pickle)AuthAWS IAM Role deloitte-ec2-role — no hardcoded credentialsInfrastructureAWS S3, Glue, SageMaker, Bedrock, Athena, EC2

Project Structure
deloitte-decision-intelligence/
├── README.md
├── frontend/
│   ├── package.json
│   └── src/
│       └── App.js              # Full React UI
├── backend/
│   ├── main.py                 # FastAPI backend + ML inference + Bedrock calls
│   └── requirements.txt        # Python dependencies
└── docs/
    ├── Team_29_Report.pdf              # Full technical report
    ├── Team29_Capstone_Poster.pdf      # Capstone poster
    └── Team29_Capstone_Final_Prez_Slides.pptx

ML Model
Ensemble Architecture
A soft-voting ensemble of three gradient-boosted tree models:

GBM — scikit-learn GradientBoostingClassifier; strong baseline on tabular data
XGBoost — adds L1/L2 regularization; handles missing values natively
LightGBM — leaf-wise tree growth; effective on large datasets

Performance

Directional Accuracy: 75.2% on held-out test set
Training Data: 120,000 synthetic business decisions
Train/Test Split: 80/20

Feature Set (22 features)
Company Profile

Industry (Manufacturing, SaaS, Retail, Healthcare, Finance)
Company Size (Small, Medium, Large)
Operating Margin
Baseline Quarterly Revenue

Decision Parameters

Decision Type (Marketing, Pricing, Expansion, R&D Investment, Hiring)
Investment Cost
Time Horizon (months)
Campaign Intensity (Low, Medium, High)

Macroeconomic Indicators

GDP Growth Rate
Inflation Expectation
Unemployment Rate
Interest Rate
30-Day Market Volatility
Market Return
VIX Index

Engineered Features (derived at inference time)

investment_ratio = Investment Cost / Revenue
macro_stress = weighted composite of VIX (40%), Unemployment (30%), negative GDP (30%)
vol_vix = 30-Day Volatility × VIX
margin_x_investment = Operating Margin × Investment Cost

Important Note on Training Data
The model was trained on synthetic data generated to cover realistic macro distributions (sourced from Bloomberg and S&P historical data). The 75.2% accuracy figure reflects performance on a held-out split of the same synthetic dataset — not on real-world client decisions. A production deployment would require retraining on real Deloitte client engagement data.

AI Layer — Claude Integration
Signal Extraction
For each prediction request containing market news text, the backend sends a prompt to Claude Sonnet 4.5 via AWS Bedrock. Claude reads the news as a whole and returns a JSON object with three signals. This is not rule-based — there is no keyword matching or sentiment dictionary. Claude makes a judgment the same way an analyst reading a news article would.
If no news text is provided, signals default to neutral values (0, 0.5, 0) and the confidence score is penalized by 8 percentage points.
Executive Report
A separate Claude call generates a three-section report — Core Risks, Recommendations, and Optimal Timing — written for a senior executive audience with no technical background. The report is specific to every input provided and regenerated on every request.
Design Principle
Claude is deliberately kept out of all numerical computation. The ML ensemble owns every number. Claude only extracts signals from text and interprets outputs the model has already calculated. This eliminates hallucinated numeric outputs and provides a fully auditable computation path.

Confidence Score
The AI confidence score is a multi-factor composite:
ComponentWeightSourceMarket Stability (VIX)35%VIX < 20 → full weight; VIX > 30 → low weightGDP Direction30%Positive GDP → full weight; negative → reducedSignal Strength20%Absolute sentiment + competition deviation from neutralBase Prior15%Aggregate ensemble accuracy on held-out test set
Penalties:

External risk flagged: −10 percentage points
No news text provided: −8 percentage points


Setup & Deployment
Prerequisites

Python 3.9+
Node.js 18+
AWS account with Bedrock access enabled in us-east-1
IAM role with S3 and Bedrock permissions attached to your EC2 instance
Trained model pickle file uploaded to S3 at s3://deloitteml/models/roi_model.pkl

Backend
bashcd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
For background deployment on EC2:
bashnohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
Frontend
bashcd frontend
npm install
npm start
The frontend expects the backend at http://<your-ec2-ip>:8000. Update the API_URL constant in App.js before deploying.
Environment & Credentials
AWS credentials are handled entirely via the EC2 IAM role (deloitte-ec2-role). No API keys or credentials should ever be hardcoded or committed to this repository.

API Reference
GET /
Health check.
json{ "status": "ok", "message": "Deloitte ROI Decision System Running" }
POST /predict
Main prediction endpoint. Accepts a JSON body with all company, decision, and macro inputs plus optional news_text. Returns predicted ROI, confidence score, risk level, investment verdict, AI signals, executive report, ROI breakdown, key factors, and industry benchmark.
Key response fields:
FieldTypeDescriptionroifloatFinal predicted ROI after ML + AI adjustmentsconfidencefloatAI confidence score (0–1)risk_levelstringLow / Medium / Medium-High / HighrecommendationstringProceed / Proceed with Caution / Do Not Proceedconsumer_sentimentfloatClaude-extracted sentiment (−1 to +1)competitive_pressurefloatClaude-extracted competition score (0–1)external_riskintClaude-extracted risk flag (0 or 1)ai_reportstringFull executive intelligence report from Clauderoi_breakdowndictLine-item breakdown of base ROI and each signal adjustmentkey_factorslistInput variables that crossed significance thresholdsbenchmarkdictIndustry average, top quartile, bottom quartile ROI

Limitations

Synthetic training data: Model has not been validated on real Deloitte client outcomes. The 75.2% accuracy figure should be treated as a proof-of-concept result.
5 industries only: Manufacturing, SaaS, Retail, Healthcare, Finance. Edge cases outside this scope will produce less reliable outputs.
Output scaling: A compression factor of 0.06 was applied empirically to bring raw model outputs into a realistic range. This has not been validated against real ROI distributions.
LLM extraction is not auditable: Claude's signal extraction cannot be verified the way a deterministic algorithm can. Outputs look structurally valid even when the interpretation is subtly wrong.
Human review required: Every output requires human review before informing any real capital allocation decision. The platform is a decision support tool, not a decision engine.


Recommended Next Steps

Validate against real client data — this is the gating condition for any client-facing deployment
A/B test the GenAI interpretation layer — measure whether narrative explanations actually reduce executive forecast overrides
Industry-level accuracy breakdown — the 75.2% aggregate may mask underperformance in specific sectors
Integrate internal data — CRM, inventory, and override history would make outputs company-specific
Harden infrastructure — containerize backend, add load balancing, auto-scaling, and model versioning for production


License & IP
This platform was developed as part of an academic capstone engagement between Carnegie Mellon University and Deloitte. Before any commercial deployment, IP ownership should be clarified in accordance with the EPA and NDA signed at the start of the engagement. See your institution's technology transfer office for guidance.
