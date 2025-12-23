# ScreenHealth AI - Digital Wellness Analysis System

AI-powered system for predicting mental health impacts based on screen time usage patterns.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation & Setup

1. **Clone/Download** this repository to your local machine

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Python virtual environment (recommended):**
   ```bash
   cd backend
   python -m venv venv
   ```

4. **Initialize database:**
   ```bash
   cd backend
   python -c "from database import db; print('Database initialized!')"
   ```

5. **Run the system:**
   ```bash
   # Terminal 1 - Backend API
   cd backend
   python app.py
   
   # Terminal 2 - Frontend Server (in new terminal)
   python -m http.server 8000
   ```

6. **Access the application:**
   - Frontend: http://localhost:8000/frontend/index.html
   - Backend API: http://localhost:5000

## 📁 Project Structure

```
├── backend/
│   ├── app.py              # Flask API server
│   ├── database.py         # SQLite database operations
│   ├── db_manager.py       # Database management script
│   └── venv/               # Python virtual environment
├── frontend/
│   ├── index.html          # Main assessment page
│   ├── assessment.html     # Screen time questionnaire
│   └── results.html        # Results and recommendations
├── models/
│   ├── mental_health_predictor.pkl  # Trained ML model
│   ├── feature_scaler.pkl           # Data preprocessing
│   ├── label_encoders.pkl           # Category encoders
│   └── feature_names.pkl            # Model features
├── data/
│   └── screen_time_survey_500.csv   # Training dataset
├── notebooks/
│   └── screentime_mental_health.ipynb  # Model development
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version for deployment
├── update_deployment_urls.py  # URL update script for deployment
├── DEPLOYMENT.md          # Deployment guide for Render
├── screenhealth.db        # SQLite database (created automatically)
└── .env                   # Environment variables
```

## 🔧 System Components

### Backend (Flask API)
- **Port:** 5000
- **Database:** SQLite (screenhealth.db)
- **Main endpoints:**
  - `GET /` - API documentation
  - `POST /api/predict` - Mental health prediction (saves to DB)
  - `POST /api/register` - User registration
  - `GET /api/user/<email>/history` - User assessment history
  - `GET /api/user/<email>/stats` - User statistics
  - `GET /api/admin/stats` - System statistics

### Frontend (HTML/JS)
- **Port:** 8000
- Interactive questionnaire for screen time assessment
- Real-time results visualization with charts
- User registration and history tracking

### Database (SQLite)
- **File:** screenhealth.db (created automatically)
- **Tables:** users, assessments, weekly_summaries, email_logs
- **Features:** User management, assessment history, email preferences

### Machine Learning Model
- Trained on 500+ user responses
- Predicts mental health status based on screen usage patterns
- Includes risk scoring and confidence metrics

## 🎯 Features

- **User Authentication:** Secure login/register system with session management
- **Personal Dashboard:** Track progress, view statistics, and manage assessments
- **AI-Powered Analysis:** Machine learning model predicts mental health impacts
- **User Management:** Registration, login, and profile management with email tracking
- **Assessment History:** Track progress over time with stored assessments and analytics
- **Personalized Recommendations:** Tailored advice based on usage patterns and history
- **Risk Assessment:** Comprehensive scoring across multiple wellness dimensions
- **Email Reports:** Weekly summary emails (configurable)
- **Interactive Dashboard:** Real-time charts and visualizations with user progress
- **Database Storage:** SQLite database for persistent data storage with user sessions
- **User Statistics:** Personal analytics, progress tracking, and goal setting
- **Professional UI:** Clean, responsive design with user-friendly navigation

## 📊 Usage

1. **Register/Login:** Create an account or login at `/login.html` for full tracking
2. **Dashboard:** View your personal dashboard with statistics and progress
3. **Take Assessment:** Complete the screen time questionnaire (logged-in users get automatic tracking)
4. **Get Prediction:** AI analyzes your patterns and predicts health status
5. **View Results:** See detailed risk scores and wellness metrics
6. **Follow Recommendations:** Get personalized tips for digital wellness
7. **Track Progress:** View your assessment history and improvement trends in dashboard
8. **Subscribe:** Enable weekly email summaries for ongoing support

## 🔐 Professional Features

### **User Authentication System**
- Secure registration and login
- Session management with localStorage
- Password protection (6+ characters required)
- Automatic user tracking across assessments

### **Personal Dashboard**
- User statistics and progress tracking
- Assessment history visualization
- Recent activity timeline
- Quick access to new assessments and reports

### **Enhanced Data Tracking**
- Every assessment automatically linked to user account
- Progress analytics over time
- Login activity tracking
- Email engagement metrics

### **Professional UI/UX**
- Clean, modern interface design
- Responsive layout for all devices
- Intuitive navigation with user context
- Professional branding and styling

## 🗄️ Database Management

### View Database Statistics
```bash
cd backend
python db_manager.py
```

### Manual Database Operations
```python
# In Python shell
from backend.database import db

# Get user stats
user = db.get_user_by_email("user@example.com")
stats = db.get_user_stats(user['id'])

# Get system stats
system_stats = db.get_database_stats()
```

### Database Schema
- **users:** User profiles and preferences
- **assessments:** All assessment results and predictions
- **weekly_summaries:** Weekly progress summaries
- **email_logs:** Email sending history

## ⚙️ Configuration

### Email Setup (Optional)
To enable weekly email reports, configure SMTP settings in `backend/app.py`:
```python
smtp_server = "your-smtp-server.com"
sender_email = "your-email@domain.com"
sender_password = "your-app-password"
```

### Environment Variables
Create `.env` file for sensitive configurations:
```
SMTP_SERVER=smtp.gmail.com
SMTP_EMAIL=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## 🛠️ Troubleshooting

**Backend won't start:**
- Ensure Python 3.8+ is installed
- Install requirements: `pip install -r requirements.txt`
- Check if port 5000 is available

**Frontend not loading:**
- Ensure Python HTTP server is running on port 8000
- Try: `python -m http.server 8000`

**Model predictions failing:**
- Check if model files exist in `/models/` directory
- System will use fallback logic if ML models are missing

## 🚀 Deployment

### Deploy to Render (Recommended)

For production deployment to make your app accessible to others:

1. **Follow the deployment guide:**
   ```bash
   # Read the comprehensive deployment guide
   cat DEPLOYMENT.md
   ```

2. **Deploy to Render.com:**
   - Create account at render.com
   - Connect your GitHub repository
   - Follow the step-by-step guide in `DEPLOYMENT.md`

3. **Update URLs after deployment:**
   ```bash
   # Update frontend URLs to point to your deployed backend
   python update_deployment_urls.py
   ```

4. **Your app will be live at:**
   - `https://your-app-name.onrender.com`

See `DEPLOYMENT.md` for detailed instructions and troubleshooting.

## 🔒 Privacy & Security

- No personal data is stored permanently
- All processing happens locally on your machine
- Email functionality is optional and configurable
- Assessment data is only used for immediate predictions

## 📈 Technical Details

- **Backend:** Flask (Python)
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **ML Framework:** scikit-learn
- **Data Processing:** pandas, numpy
- **Visualization:** Chart.js

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify both backend and frontend servers are running

---

**Note:** This system is for educational and wellness awareness purposes. For serious mental health concerns, please consult healthcare professionals.