# 📱 Telco Customer Churn Prediction App

A production-ready Streamlit application for predicting customer churn in telecommunications companies using machine learning.

**Developed by:** TS Academy Group 9  
**Track:** Classification  
**Dataset:** Telco Customer Churn (IBM)

---

## 🎯 Features

- **Interactive Prediction Interface**: User-friendly sidebar for customer data input
- **Dual Model Support**: 
  - Balanced Logistic Regression (78.34% recall, optimized for cost)
  - Random Forest (62.56% F1-score)
- **Cost-Optimized Predictions**: Uses threshold of 0.29 to minimize revenue loss
- **Visual Analytics**: 
  - Probability gauge charts
  - Risk factor analysis
  - Detailed recommendations
- **Downloadable Reports**: Export predictions as CSV
- **Mobile Responsive**: Works on all devices

---

## 🚀 Quick Start (Local Deployment)

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Setup Project

```bash
# Create project directory
mkdir telco-churn-app
cd telco-churn-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get the Dataset

Download the Telco Customer Churn dataset:

**Option A: Kaggle**
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Place it in your project directory

**Option B: IBM Community**
1. Go to: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
2. Download the dataset
3. Save as `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Step 3: Train Models

```bash
python train_model.py
```

This will create:
- `logistic_model.pkl` (Recommended model)
- `rf_model.pkl` (Random Forest model)
- `model_metadata.pkl` (Training metadata)

### Step 4: Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ☁️ Deploy to Streamlit Community Cloud (FREE)

### Prerequisites
- GitHub account
- Streamlit account (free, sign up with GitHub)

### Step-by-Step Deployment

#### 1. Prepare Your Repository

```bash
# Initialize git (if not already done)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit - Telco Churn Prediction App"

# Create repository on GitHub (via web interface)
# Then link and push:
git remote add origin https://github.com/YOUR_USERNAME/telco-churn-app.git
git branch -M main
git push -u origin main
```

#### 2. Important: Upload Pre-trained Models

Since the dataset is large, you should:

**Option A: Include models in repo (if < 100MB)**
- After running `train_model.py`, commit the `.pkl` files:
```bash
git add logistic_model.pkl rf_model.pkl model_metadata.pkl
git commit -m "Add trained models"
git push
```

**Option B: Use Git LFS for large files**
```bash
# Install Git LFS
git lfs install

# Track pickle files
git lfs track "*.pkl"
git add .gitattributes
git add *.pkl
git commit -m "Add models with LFS"
git push
```

#### 3. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **"New app"**
3. Sign in with GitHub
4. Configure deployment:
   - **Repository**: Select your `telco-churn-app` repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

Your app will be live at: `https://YOUR_USERNAME-telco-churn-app.streamlit.app`

#### 4. Verify Deployment

After deployment:
- ✅ Check that all models load correctly
- ✅ Test predictions with sample data
- ✅ Verify visualizations render properly

---

## 📁 Project Structure

```
telco-churn-app/
├── app.py                           # Main Streamlit application
├── train_model.py                   # Model training script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (not in git)
├── logistic_model.pkl              # Trained LR model
├── rf_model.pkl                    # Trained RF model
└── model_metadata.pkl              # Training metadata
```

---

## 🔧 Troubleshooting

### Issue: Models not loading

**Error:** `Model files not found`

**Solution:**
1. Ensure you ran `python train_model.py` successfully
2. Check that `.pkl` files exist in the project directory
3. If deploying to Streamlit Cloud, ensure models are committed to git

### Issue: Dataset not found during training

**Error:** `Dataset not found at WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Solution:**
1. Download the dataset from Kaggle or IBM Community
2. Place it in the project root directory
3. Ensure the filename is exactly: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

### Issue: Streamlit Cloud deployment fails

**Solution:**
1. Check the logs in Streamlit Cloud dashboard
2. Ensure all dependencies are in `requirements.txt`
3. Verify Python version compatibility (use 3.8-3.11)
4. For large model files, use Git LFS

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## 📊 Model Performance

### Balanced Logistic Regression (Recommended)
- **Accuracy**: 78.55%
- **Precision**: 61.36%
- **Recall**: 78.34%
- **F1-Score**: 61.36%
- **Optimized Threshold**: 0.29
- **Financial Impact**: Saves ~₦775,000 in retention costs

### Random Forest
- **Accuracy**: 78.28%
- **F1-Score**: 62.56%
- **Best for**: Capturing non-linear patterns

---

## 🎓 Key Insights from Analysis

1. **High-Risk Indicators:**
   - Month-to-month contracts
   - Fiber optic internet service
   - Electronic check payment method
   - New customers (tenure < 12 months)
   - High monthly charges (>$70)

2. **Protective Factors:**
   - Long tenure (>38 months)
   - Long-term contracts (1-2 years)
   - Automatic payment methods
   - Tech support subscription
   - Online security services

3. **Churn Rate:**
   - Baseline: 26.54%
   - Target: Reduce to <20% through proactive retention

---

## 🛠️ Customization

### Change Model Threshold

In `app.py`, modify:
```python
optimized_threshold = 0.29  # Adjust based on your cost analysis
```

### Add New Features

1. Update `train_model.py` with new feature columns
2. Retrain models: `python train_model.py`
3. Update input fields in `app.py` sidebar

### Modify UI Theme

In `app.py`, adjust custom CSS:
```python
st.markdown("""
    <style>
    .main-header {
        color: #YOUR_COLOR;  # Change colors
    }
    </style>
    """, unsafe_allow_html=True)
```

---

## 📈 Future Enhancements

- [ ] A/B testing for different thresholds
- [ ] Integration with CRM systems
- [ ] Batch prediction upload (CSV)
- [ ] Historical trend analysis
- [ ] Real-time model retraining
- [ ] Multi-language support

---

## 👥 Team Members

| Name | Email | GitHub |
|------|-------|--------|
| Jemiri Daniel Taiwo | updatedan2@gmail.com | [@jemiridaniel](https://github.com/jemiridaniel) |
| Kuram John Sokomba | Sokomba16@gmail.com | [@SokombaGit](https://github.com/SokombaGit) |
| Oluba Amos Oluwasegun | amosoluba@gmail.com | [@Famous-Amos1](https://github.com/Famous-Amos1) |
| Mbata Chidumaga | mbataechidumaga@gmail.com | [@chidumaga](https://github.com/chidumaga) |
| Bethel Ya'u | sheshamkaza@gmail.com | [@She-Wins](https://github.com/She-Wins) |
| Nwafor Deborah | nwafordeborah41@gmail.com | [@DRDEBBIE256](https://github.com/DRDEBBIE256) |
| Jeremiah Yusuf | jeremiahyusuf185@gmail.com | [@Jeremy-1020](https://github.com/Jeremy-1020) |
| Pyagbara Prince | princepyagbara@gmail.com | [@PrinceZorzor](https://github.com/PrinceZorzor) |

---

## 📝 License

This project is developed as part of TS Academy Capstone Project.

---

## 🙏 Acknowledgments

- **TS Academy** for the learning opportunity
- **IBM** for the Telco Customer Churn dataset
- **Streamlit** for the amazing framework

---

## 📧 Support

For questions or issues:
1. Check the Troubleshooting section above
2. Open an issue on GitHub
3. Contact any team member via email

---

**Happy Predicting! 🎯**
