# Employee Performance Prediction System 

## Overview
A machine learning solution built to predict employee performance ratings at INX Future Inc. using historical HR data. The system provides accurate performance predictions, identifies key performance drivers, and delivers actionable business recommendations. Developed using Python, Scikit-learn, and Streamlit, the model achieves **85.4% accuracy** with strong interpretability and reliability.

## Key Highlights
- **Random Forest Classifier** achieving **85.4% accuracy** and **87.3% prediction confidence**  
- Analyzed **42+ employee attributes** to identify top performance drivers  
- Automated recommendation engine for personalized employee development  
- Real-time, interactive **Streamlit dashboard** for prediction and visualization  
- Business ROI potential estimated at **$2.8M annually**

## Technical Summary
- **Algorithm:** Random Forest (balanced class weights, 100 estimators, Gini criterion)  
- **Data Processing:** One-hot encoding, standard scaling, feature engineering  
- **Validation:** Stratified Train/Test Split + 5-Fold Cross-Validation (84.2% ± 2.1%)  
- **Feature Importance:** Key drivers include Environment Satisfaction, Job Satisfaction, Promotion timelines, Work-Life Balance, Tenure, and Training frequency  
- **Deployment:** Streamlit web app + joblib model saving

## Business Insights
- Employees promoted within the last year exhibit reduced performance due to adjustment periods  
- Training benefits plateau after 3 sessions/year  
- Overtime correlates with **lower performance (−16%)**, contrary to assumptions  
- Job and environment satisfaction below 2.5 sharply reduce performance  
- Education level shows weak correlation with performance

## Recommendations
- Implement monthly predictive performance monitoring  
- Address career stagnation for employees with 3+ years since last promotion  
- Launch environment satisfaction initiatives  
- Reduce overtime via workload redistribution  
- Standardize 2 high-impact training sessions annually  
- Strengthen manager capability through coaching programs

## Dataset Summary
- **1200 employee records (2020–2023)**  
- **42+ features**, performance rating scale from 1 to 5  
- Class imbalance managed through class weighting  
- Rich data covering satisfaction, career growth, compensation, role, experience, and work characteristics

## Project Structure
- `/data` — raw, processed, engineered features  
- `/models` — trained ML models  
- `/src` — preprocessing, feature engineering, model training scripts  
- `/notebooks` — EDA, engineering, and training notebooks  
- `app.py` — Streamlit dashboard  
- `requirements.txt`, `README.md`, `LICENSE`

## Impact Targets (Next 12 Months)
- Increase overall performance rating from **3.4 → 3.7**  
- Reduce low performers from **18% → <10%**  
- Improve engagement scores by **18%+**  
- Reduce voluntary attrition from **19% → <12%**

