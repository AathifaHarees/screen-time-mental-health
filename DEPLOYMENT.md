# 🚀 **ScreenHealth AI - Deployment Guide**

## **🎯 Quick Deployment (3 Minutes)**

### **Step 1: Push Latest Changes**
```bash
# Push your latest changes to your existing repo
git add .
git commit -m "Ready for deployment - cleaned up files"
git push origin main
```

### **Step 2: Deploy to Render**
1. Go to **[render.com](https://render.com)** and sign up with GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect your existing repository: **`screen-time-mental-health`**
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd backend && python app.py`
   - **Environment Variable**: `FLASK_ENV=production`
5. Click **"Create Web Service"**

### **Step 3: Update URLs**
```bash
# Update frontend to use your live URL
python update_deployment_urls.py
# Enter: https://your-app-name.onrender.com

git add .
git commit -m "Update URLs for deployment"
git push origin main
```

**🎉 Your app is live at:** `https://your-app-name.onrender.com`

---

## **✅ Why Render?**
- **🆓 Always On**: No sleep mode (unlike Heroku free)
- **⚡ Fast**: Global CDN, 2-3 minute deployments
- **🔄 Auto-Deploy**: Push to GitHub = automatic deployment
- **🔒 Secure**: Free HTTPS, perfect for healthcare apps

---

## **📱 What Users Get**
- ✅ Professional digital wellness assessments
- ✅ AI-powered health analysis
- ✅ Medical-grade PDF reports
- ✅ Mobile-friendly interface
- ✅ Secure user accounts and data

---

## **🆘 Need Help?**
- **Build fails**: Check Python version in `runtime.txt`
- **App won't start**: Verify `cd backend && python app.py` command
- **Frontend errors**: Make sure URLs are updated correctly

**Total deployment time: 3-5 minutes!** �

---

## **📋 Your Repository**
- **GitHub**: https://github.com/AathifaHarees/screen-time-mental-health.git
- **Ready to deploy**: All files cleaned up and configured
- **Next step**: Just push changes and deploy to Render!