# 🚀 STREAMLIT DEPLOYMENT CHECKLIST

Follow these steps to deploy your telco churn app for FREE on Streamlit Community Cloud.

---

## ✅ PRE-DEPLOYMENT CHECKLIST

### 1. Files Ready
- [ ] `app.py` exists
- [ ] `train_model.py` exists
- [ ] `requirements.txt` exists
- [ ] `README.md` exists
- [ ] `.gitignore` exists

### 2. Dataset & Models
- [ ] Downloaded `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- [ ] Ran `python train_model.py` successfully
- [ ] `logistic_model.pkl` created (check file size)
- [ ] `rf_model.pkl` created (check file size)
- [ ] `model_metadata.pkl` created

### 3. Local Testing
- [ ] Installed dependencies: `pip install -r requirements.txt`
- [ ] App runs locally: `streamlit run app.py`
- [ ] Tested prediction with sample data
- [ ] All visualizations render correctly
- [ ] No errors in console

---

## 📤 GITHUB SETUP

### Step 1: Create Repository
```bash
# Initialize git
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Telco Churn Prediction App"
```

### Step 2: Create GitHub Repo
1. Go to https://github.com/new
2. Repository name: `telco-churn-app` (or your choice)
3. Description: "Telco Customer Churn Prediction with ML"
4. Make it **Public** (required for free Streamlit hosting)
5. Don't initialize with README (we already have one)
6. Click **Create repository**

### Step 3: Push to GitHub
```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/telco-churn-app.git

# Push
git branch -M main
git push -u origin main
```

### Step 4: Verify Upload
- [ ] Visit your GitHub repo URL
- [ ] Confirm all files are visible
- [ ] Check that `.pkl` files are uploaded (should be in commits)

---

## ☁️ STREAMLIT CLOUD DEPLOYMENT

### Step 1: Sign Up
1. Go to https://share.streamlit.io
2. Click **"Sign up"**
3. Use **"Continue with GitHub"**
4. Authorize Streamlit

### Step 2: Deploy App
1. Click **"New app"** button
2. Fill in deployment settings:
   - **Repository**: `YOUR_USERNAME/telco-churn-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose your subdomain (e.g., `telco-churn-predictor`)

3. Click **"Deploy!"**

### Step 3: Wait for Build
- [ ] Build starts (watch logs)
- [ ] Dependencies install (takes 2-3 minutes)
- [ ] App initializes
- [ ] Status shows "Running"

### Step 4: Test Deployed App
- [ ] Visit your app URL: `https://YOUR-SUBDOMAIN.streamlit.app`
- [ ] Enter sample customer data
- [ ] Click "Predict Churn Risk"
- [ ] Verify prediction displays correctly
- [ ] Check all visualizations work
- [ ] Download report feature works

---

## 🐛 TROUBLESHOOTING

### Build Fails with "ModuleNotFoundError"
**Fix:** Add missing module to `requirements.txt`
```bash
# Update requirements.txt, then:
git add requirements.txt
git commit -m "Update dependencies"
git push
# Streamlit will auto-redeploy
```

### Models Not Loading
**Error:** "Model files not found"

**Fix:**
1. Check if `.pkl` files are in your GitHub repo
2. If files are too large (>100MB), use Git LFS:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add *.pkl
git commit -m "Add models with LFS"
git push
```

### App Stuck on "Running"
**Fix:** Check Streamlit Cloud logs for errors
1. Go to your app dashboard
2. Click "Manage app"
3. View logs
4. Fix any Python errors shown

---

## 🎉 POST-DEPLOYMENT

### Share Your App
- [ ] Get shareable URL: `https://YOUR-SUBDOMAIN.streamlit.app`
- [ ] Add URL to GitHub README
- [ ] Share with team/stakeholders
- [ ] Add to your portfolio

### Monitor Usage
- [ ] Check Streamlit Cloud dashboard for analytics
- [ ] Monitor app performance
- [ ] Review user feedback

### Update App
To make changes:
```bash
# Edit files locally
# Test: streamlit run app.py
git add .
git commit -m "Describe your changes"
git push
# Streamlit auto-redeploys!
```

---

## 📊 ESTIMATED TIMELINE

| Task | Duration |
|------|----------|
| Setup local environment | 5 minutes |
| Download dataset | 2 minutes |
| Train models | 3-5 minutes |
| Test locally | 5 minutes |
| Create GitHub repo | 3 minutes |
| Push to GitHub | 2 minutes |
| Deploy on Streamlit | 5-7 minutes |
| **TOTAL** | **25-30 minutes** |

---

## 🆘 NEED HELP?

### Resources
- Streamlit Docs: https://docs.streamlit.io
- Streamlit Forum: https://discuss.streamlit.io
- Deployment Guide: https://docs.streamlit.io/streamlit-community-cloud

### Common Issues
1. **"Repository not found"** → Make repo public
2. **"File not found"** → Check file paths in app.py
3. **"Module not found"** → Update requirements.txt
4. **"Model error"** → Retrain models, verify .pkl files

---

## ✨ SUCCESS CRITERIA

Your deployment is successful when:
- ✅ App URL is accessible
- ✅ No error messages
- ✅ Predictions work correctly
- ✅ All visualizations display
- ✅ Download report works

---

**You're ready to deploy! 🚀**

Start with the PRE-DEPLOYMENT CHECKLIST above and follow each section in order.
