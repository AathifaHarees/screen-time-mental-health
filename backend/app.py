"""
Screen Time Mental Health Prediction System
Part 3: Flask Backend API (Production-Ready)
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import json
import os


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ==================== LOAD ML MODEL & PREPROCESSING ====================

class ScreenTimePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessing objects"""
        try:
            self.model = joblib.load('mental_health_predictor.pkl')
            self.scaler = joblib.load('feature_scaler.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.feature_names = joblib.load('feature_names.pkl')
            print("✓ Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Using fallback prediction logic")
    
    def preprocess_input(self, user_data):
        """Preprocess user input to match training data format"""
        
        # Ordinal mappings
        ordinal_mappings = {
            'Average total screen time per day': {
                'Less than 1 hour': 0, '1–2 hours': 1, '3–4 hours': 2, 
                '5–6 hours': 3, 'More than 6 hours': 4
            },
            'Social Media (hours)': {
                'Less than 1 hour': 0, '1–2 hours': 1, '3–4 hours': 2, 
                '5–6 hours': 3, 'More than 6 hours': 4
            },
            'Entertainment (hours)': {
                'Less than 1 hour': 0, '1–2 hours': 1, '3–4 hours': 2, 
                '5–6 hours': 3, 'More than 6 hours': 4
            },
            'Average sleep duration (hours)': {
                'Less than 5': 0, '5–6': 1, '6–7': 2, '7–8': 3, 'More than 8': 4
            },
            'Use screen before sleep': {
                'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
            },
            'Exercise frequency': {
                'Never': 0, '1–2 days per week': 1, '3–4 days per week': 2, 'Daily': 3
            },
            'Feel addicted to devices': {
                'No': 0, 'Not sure': 1, 'Maybe': 2, 'Yes': 3
            }
        }
        
        processed_data = {}
        
        # Apply ordinal encoding
        for key, value in user_data.items():
            if key in ordinal_mappings:
                processed_data[key] = ordinal_mappings[key].get(value, 0)
            elif isinstance(value, str) and key in self.label_encoders:
                try:
                    processed_data[key] = self.label_encoders[key].transform([value])[0]
                except:
                    processed_data[key] = 0
            else:
                processed_data[key] = value
        
        return processed_data
    
    def calculate_risk_scores(self, processed_data):
        """Calculate various risk scores"""
        
        # Screen time risk
        screen_time_score = (
            processed_data.get('Average total screen time per day', 0) * 5 +
            processed_data.get('Social Media (hours)', 0) * 3 +
            processed_data.get('Entertainment (hours)', 0) * 2
        )
        
        # Sleep risk
        sleep_risk = (
            (4 - processed_data.get('Average sleep duration (hours)', 0)) * 3 +
            (5 - processed_data.get('Sleep quality (1–5)', 3)) * 2 +
            processed_data.get('Use screen before sleep', 0) * 2
        )
        
        # Behavioral risk
        behavioral_risk = (
            processed_data.get('Stress due to screen usage', 0) * 2 +
            processed_data.get('Anxious without device', 0) * 2 +
            processed_data.get('Eye strain or headache', 0) +
            processed_data.get('Feel addicted to devices', 0) * 2
        )
        
        # Physical activity benefit
        exercise_benefit = (3 - processed_data.get('Exercise frequency', 0)) * 2
        
        total_risk = screen_time_score + sleep_risk + behavioral_risk + exercise_benefit
        
        return {
            'total_risk': total_risk,
            'screen_time_score': screen_time_score,
            'sleep_risk': sleep_risk,
            'behavioral_risk': behavioral_risk,
            'screen_index': min(100, total_risk * 1.5),
            'sleep_health': max(0, 100 - sleep_risk * 3),
            'wellbeing_score': max(0, 100 - behavioral_risk * 2.5)
        }
    
    def predict(self, user_data):
        """Make prediction using ML model or fallback logic"""
        
        processed_data = self.preprocess_input(user_data)
        risk_scores = self.calculate_risk_scores(processed_data)
        
        # Use ML model if available
        if self.model is not None:
            try:
                # Create feature vector matching training data
                feature_vector = pd.DataFrame([processed_data])
                feature_vector = feature_vector.reindex(columns=self.feature_names, fill_value=0)
                
                # Scale features
                feature_vector_scaled = self.scaler.transform(feature_vector)
                
                # Predict
                prediction = self.model.predict(feature_vector_scaled)[0]
                probability = self.model.predict_proba(feature_vector_scaled)[0]
                
                is_healthy = prediction == 1
                confidence = max(probability)
                
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fallback to rule-based prediction
                is_healthy = risk_scores['total_risk'] < 50
                confidence = 0.75
        else:
            # Rule-based prediction
            is_healthy = risk_scores['total_risk'] < 50
            confidence = 0.75
        
        return {
            'is_healthy': bool(is_healthy),
            'confidence': float(confidence),
            'risk_scores': risk_scores,
            'processed_data': processed_data
        }

# Initialize predictor
predictor = ScreenTimePredictor()

# ==================== RECOMMENDATION ENGINE ====================

class RecommendationEngine:
    """Generate personalized recommendations"""
    
    @staticmethod
    def generate_recommendations(prediction, user_data):
        """Create personalized recommendations based on risk profile and user data"""
        
        risk_scores = prediction.get('risk_scores', {})
        
        recommendations = {
            'wellness_tips': [],
            'app_lock_suggestions': [],
            'meditation_exercises': [],
            'parental_controls': []
        }
        
        # ==================== WELLNESS TIPS ====================
        total_screen = user_data.get('totalScreenTime', '')
        stress_level = int(user_data.get('stress', 3))
        sleep_before_screen = user_data.get('screenBeforeSleep', '')
        
        # Personalized based on actual screen time
        if total_screen == 'More than 6 hours':
            recommendations['wellness_tips'].extend([
                "🚨 CRITICAL: Your 6+ hours of daily screen time is excessive and harmful",
                "⏰ URGENT: Take a 10-minute break EVERY 30 minutes",
                "👁️ MANDATORY: Practice 20-20-20 rule - Every 20 min, look 20 feet away for 20 sec",
                "🚫 Create strict device-free zones (bedroom, bathroom, dining area)",
                "📵 Set your phone to grayscale mode after 6 PM",
                "💪 Target: Reduce screen time by 2 hours this week"
            ])
        elif total_screen == '5–6 hours':
            recommendations['wellness_tips'].extend([
                "⚠️ Your 5-6 hours of daily screen time is above healthy limits",
                "⏰ Take 5-minute breaks every hour",
                "👁️ Practice 20-20-20 rule for eye health",
                "🚫 Establish device-free zones in your home",
                "🌅 Implement 'Digital Sunset' - no screens 1 hour before bed",
                "📊 Target: Reduce to under 4 hours daily"
            ])
        elif total_screen == '3–4 hours':
            recommendations['wellness_tips'].extend([
                "✅ Your screen time is moderate but can improve",
                "⏰ Continue taking hourly breaks",
                "👁️ Use blue light filters in the evening",
                "📊 Track your usage to maintain awareness",
                "🎯 Try to keep it under 4 hours"
            ])
        elif total_screen == '1–2 hours':
            recommendations['wellness_tips'].extend([
                "🌟 Great! Your screen time is within healthy limits",
                "✅ Keep maintaining these excellent habits",
                "📊 Continue tracking to stay accountable",
                "💡 Consider sharing your strategies with others"
            ])
        else:  # Less than 1 hour
            recommendations['wellness_tips'].extend([
                "🏆 Outstanding! Your screen time is exceptionally healthy",
                "✨ You're a role model for digital wellness",
                "📚 Consider mentoring others on healthy tech habits",
                "🎯 Maintain this excellent balance"
            ])
        
        # Add stress-specific tips
        if stress_level >= 4:
            recommendations['wellness_tips'].insert(1, f"😰 High stress level ({stress_level}/5) detected - Screen breaks are crucial for your mental health")
        
        # Add sleep-specific tips
        if sleep_before_screen in ['Always', 'Often']:
            recommendations['wellness_tips'].append("😴 WARNING: Using screens before sleep is severely harming your rest - Stop 1-2 hours before bed")
        
        # ==================== APP LOCK SUGGESTIONS ====================
        social_media = user_data.get('socialMedia', '')
        entertainment = user_data.get('entertainment', '')
        
        if social_media == 'More than 6 hours':
            recommendations['app_lock_suggestions'].extend([
                "📱 EMERGENCY: 6+ hours on social media is extremely harmful to mental health",
                "🔒 IMMEDIATELY set 1-hour daily limit on ALL social apps",
                "⏱️ Block Instagram, TikTok, Facebook completely after limit reached",
                "⚫ Force grayscale mode on your phone permanently",
                "🔔 DELETE social media apps for a 1-week digital detox",
                "📅 Allow access ONLY during 12-1 PM, no exceptions"
            ])
        elif social_media == '5–6 hours':
            recommendations['app_lock_suggestions'].extend([
                "📱 CRITICAL: 5-6 hours on social media is excessive",
                "⏱️ Set strict 2-hour daily limit on social apps NOW",
                "🔒 Enable app timers: Instagram (30min), TikTok (30min), Twitter (30min)",
                "⚫ Switch to grayscale after 8 PM",
                "🔔 Disable ALL social media notifications immediately",
                "📅 Schedule specific windows: 12-1 PM and 6-7 PM only"
            ])
        elif social_media == '3–4 hours':
            recommendations['app_lock_suggestions'].extend([
                "📱 3-4 hours on social media is above healthy limits",
                "⏱️ Reduce to 2 hours daily using app timers",
                "🔔 Turn off all non-essential notifications",
                "⚫ Use grayscale mode after 9 PM",
                "🎯 Set a 90-minute daily limit this week"
            ])
        elif social_media == '1–2 hours':
            recommendations['app_lock_suggestions'].extend([
                "✅ Your 1-2 hours of social media is reasonable",
                "📱 Maintain these current healthy limits",
                "🎯 Consider reducing to under 1 hour for even better well-being",
                "🔔 Keep notifications minimal"
            ])
        else:  # Less than 1 hour
            recommendations['app_lock_suggestions'].extend([
                "🌟 Excellent! Your social media usage is exemplary",
                "✅ You've achieved great digital discipline",
                "💡 Share your strategies with others who struggle",
                "🎯 Maintain this healthy boundary"
            ])
        
        # Add entertainment-specific suggestions
        if entertainment in ['More than 6 hours', '5–6 hours']:
            recommendations['app_lock_suggestions'].insert(1, f"🎮 Your entertainment time ({entertainment}) is excessive - Set 2-hour daily limit")
        elif entertainment in ['3–4 hours']:
            recommendations['app_lock_suggestions'].append(f"🎬 Entertainment time ({entertainment}) can be reduced - Try 2 hours max")
        
        # In generate_recommendations function
        work_time = user_data.get('workTime', '')

        if work_time in ['More than 6 hours', '5–6 hours']:
            recommendations['wellness_tips'].extend([
                "💼 High work screen time detected - Important for ergonomics",
                "🪑 Ensure proper chair and desk setup",
                "⏰ Use Pomodoro technique: 25 min work, 5 min break",
                "👁️ Extra important to practice 20-20-20 rule"
            ])
        elif work_time in ['3–4 hours']:
            recommendations['wellness_tips'].append("💼 Moderate work screen time - Maintain good posture")

        # ==================== MEDITATION & EXERCISE ====================
        stress = int(user_data.get('stress', 3))
        anxious = int(user_data.get('anxious', 3))
        eye_strain = int(user_data.get('eyeStrain', 3))
        exercise_freq = user_data.get('exercise', '')
        sleep_quality = int(user_data.get('sleepQuality', 3))
        
        # High stress/anxiety - intensive recommendations
        if stress >= 4 or anxious >= 4:
            recommendations['meditation_exercises'].extend([
                f"🧘 DAILY REQUIREMENT: 15-minute guided meditation (Your stress: {stress}/5, Anxiety: {anxious}/5)",
                "🫁 Box breathing 4x daily: Inhale 4s → Hold 4s → Exhale 4s → Hold 4s",
                "🌙 30-minute evening yoga specifically for stress relief",
                "🚶 Two 20-minute outdoor walks WITHOUT phone - non-negotiable",
                "💆 Progressive muscle relaxation before bed (YouTube: 'PMR guided')",
                "🎵 Listen to 432Hz healing music during all screen breaks",
                "📝 5-minute anxiety journaling every evening before bed"
            ])
        elif stress >= 3 or anxious >= 3:
            recommendations['meditation_exercises'].extend([
                f"🧘 Daily 10-minute meditation recommended (Stress: {stress}/5, Anxiety: {anxious}/5)",
                "🫁 Practice box breathing during stressful moments (3-4 times daily)",
                "🌙 15-minute evening yoga or stretching routine",
                "🚶 Daily 20-minute mindful walk without devices",
                "💆 Try progressive muscle relaxation 3x per week"
            ])
        else:
            recommendations['meditation_exercises'].extend([
                f"✨ Your stress ({stress}/5) and anxiety ({anxious}/5) levels are healthy!",
                "🧘 Continue your current wellness routine",
                "🔄 Explore new meditation techniques for variety",
                "📚 Consider deepening your mindfulness practice"
            ])
        
        # Sleep quality issues
        if sleep_quality <= 2:
            recommendations['meditation_exercises'].insert(0, f"😴 CRITICAL: Sleep quality ({sleep_quality}/5) is very poor - Try bedtime yoga & sleep meditation apps")
        elif sleep_quality == 3:
            recommendations['meditation_exercises'].append(f"😴 Sleep quality ({sleep_quality}/5) needs improvement - Evening relaxation routine recommended")
        
        # Exercise recommendations
        if exercise_freq == 'Never':
            recommendations['meditation_exercises'].extend([
                "💪 URGENT: You're not exercising at all - This is critical!",
                "🏃 START TODAY: Take a 20-minute walk right now",
                "🎯 GOAL: Exercise 3-4 days per week minimum for mental health",
                "🏋️ Even 10 minutes daily makes a huge difference"
            ])
        elif exercise_freq == '1–2 days per week':
            recommendations['meditation_exercises'].insert(2, "💪 Good start! Increase to 3-4 days per week for optimal mental health")
        elif exercise_freq == '3–4 days per week':
            recommendations['meditation_exercises'].append("💪 Great exercise routine! Maintain 3-4 days per week")
        else:  # Daily
            recommendations['meditation_exercises'].append("🏆 Excellent! Daily exercise is optimal for mental health")
        
        # Eye strain
        if eye_strain >= 4:
            recommendations['meditation_exercises'].insert(0, f"👁️ SEVERE eye strain ({eye_strain}/5) - Take immediate breaks, use eye drops, reduce brightness")
        elif eye_strain >= 3:
            recommendations['meditation_exercises'].append(f"👁️ Moderate eye strain ({eye_strain}/5) - Increase breaks, adjust lighting")
        
        # ==================== ACCOUNTABILITY & CONTROLS ====================
        age_group = user_data.get('ageGroup', '')
        addicted = user_data.get('addicted', '')
        
        # Device addiction concerns
        if addicted == 'Yes':
            recommendations['parental_controls'].extend([
                "🚨 You've identified device addiction - This requires immediate action",
                "🤝 URGENT: Find an accountability partner for DAILY check-ins",
                "📞 CONSIDER: Professional help - Speak with a therapist about screen addiction",
                "🌳 Use aggressive blocking: Cold Turkey (blocks everything), Freedom app",
                "👥 JOIN NOW: r/nosurf community, Digital Detox support groups",
                "📊 Share your screen time with someone you trust EVERY single day",
                "🏆 30-day challenge: Reduce screen time by 50%",
                "💰 Accountability: Donate $10 for every day you exceed limits"
            ])
        elif addicted == 'Maybe':
            recommendations['parental_controls'].extend([
                "⚠️ You suspect device addiction - Take this seriously before it worsens",
                "🤝 Find an accountability partner for weekly check-ins",
                "🌳 Try focus apps: Forest, Freedom, Cold Turkey Blocker",
                "👥 Join digital wellness communities: r/digitalminimalism",
                "📊 Track daily and share progress weekly",
                "🏆 Set challenges with friends (lowest screen time wins rewards)"
            ])
        elif addicted == 'Not sure':
            recommendations['parental_controls'].extend([
                "🤔 Uncertain about addiction? Monitor yourself for 2 weeks",
                "📊 Track everything and assess patterns honestly",
                "🤝 Consider finding an accountability buddy",
                "👥 Join supportive communities for guidance"
            ])
        else:  # No
            recommendations['parental_controls'].extend([
                "✅ Good awareness - you don't feel addicted",
                "📊 Continue monitoring to maintain healthy habits",
                "🤝 Help others who struggle with screen addiction"
            ])
        
        # Age-specific recommendations
        if age_group in ['18–24']:
            recommendations['parental_controls'].extend([
                "👥 Build peer accountability - Share goals with friends your age",
                "💼 Replace social media time with career skill-building",
                "🎓 Use screen time for learning valuable skills (coding, languages)",
                "📱 Your age group is most vulnerable - Be extra vigilant"
            ])
        elif age_group in ['25–34']:
            recommendations['parental_controls'].extend([
                "💼 Prioritize screen time for career advancement over entertainment",
                "🤝 Create accountability groups with colleagues/friends",
                "📈 Track how screen habits affect your productivity",
                "🎯 Set professional development goals over social media"
            ])
        elif age_group in ['35–44', '45–54']:
            recommendations['parental_controls'].extend([
                "👨‍👩‍👧‍👦 MODEL healthy behavior - Your kids are watching you",
                "🍽️ Enforce device-free family meals (no phones at table)",
                "📱 Use Screen Time (iOS) or Family Link (Android) for kids",
                "📋 Create written family media agreement together",
                "🎯 Lead family screen time challenges with rewards"
            ])
        elif age_group == '55+':
            recommendations['parental_controls'].extend([
                "👨‍👩‍👧‍👦 Set example for younger generations in family",
                "🎓 Use screen time for learning and staying connected meaningfully",
                "📱 Balance technology with face-to-face interactions",
                "💡 Share your wisdom on pre-digital era balance"
            ])
        
        # Risk-based final recommendations
        total_risk = risk_scores.get('total_risk', 0)
        if total_risk > 70:
            recommendations['parental_controls'].insert(0, f"🚨🚨🚨 CRITICAL RISK ({int(total_risk)}/100) - Seek professional help immediately!")
        elif total_risk > 60:
            recommendations['parental_controls'].insert(0, f"🚨 HIGH RISK ({int(total_risk)}/100) - Immediate intervention needed, get support NOW")
        elif total_risk > 50:
            recommendations['parental_controls'].insert(0, f"⚠️ ELEVATED RISK ({int(total_risk)}/100) - Take action this week, start with accountability partner")
        
        return recommendations
        
# ==================== API ENDPOINTS ====================

@app.route('/')
def home():
    """Home page with API documentation"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Screen Time Mental Health API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; }
            h1 { color: #667eea; }
            .endpoint { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
            code { background: #333; color: #fff; padding: 2px 8px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>🧠 Screen Time Mental Health Prediction API</h1>
        <p>AI-powered system for predicting mental health impacts based on screen time usage patterns.</p>
        
        <div class="endpoint">
            <h3>POST /api/predict</h3>
            <p>Predict mental health status based on screen time and behavioral data</p>
            <p><strong>Example Request:</strong></p>
            <code>POST /api/predict</code>
            <p><strong>Body:</strong> JSON object with user data</p>
        </div>
        
        <div class="endpoint">
            <h3>POST /api/recommendations</h3>
            <p>Get personalized recommendations based on prediction</p>
        </div>
        
        <div class="endpoint">
            <h3>POST /api/send-weekly-summary</h3>
            <p>Send weekly summary email to user</p>
        </div>
        
        <div class="endpoint">
            <h3>GET /api/health</h3>
            <p>Check API health status</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        user_data = request.json
        
        # Validate input
        if not user_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        prediction = predictor.predict(user_data)
        
        # Generate recommendations
        recommendations = RecommendationEngine.generate_recommendations(prediction, user_data)
        
        # Prepare response
        response = {
            'prediction': {
                'is_healthy': prediction['is_healthy'],
                'confidence': round(prediction['confidence'], 3),
                'status': 'Healthy' if prediction['is_healthy'] else 'Needs Improvement'
            },
            'risk_scores': {
                'total_risk': round(prediction['risk_scores']['total_risk'], 2),
                'screen_index': round(prediction['risk_scores']['screen_index'], 2),
                'sleep_health': round(prediction['risk_scores']['sleep_health'], 2),
                'wellbeing_score': round(prediction['risk_scores']['wellbeing_score'], 2)
            },
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get detailed recommendations"""
    try:
        data = request.json
        prediction = data.get('prediction', {})
        user_data = data.get('user_data', {})
        
        recommendations = RecommendationEngine.generate_recommendations(prediction, user_data)
        
        return jsonify({
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-weekly-summary', methods=['POST'])
def send_weekly_summary():
    """Send weekly summary email"""
    try:
        data = request.json
        email = data.get('email')
        summary_data = data.get('summary_data', {})
        
        if not email:
            return jsonify({'error': 'Email address required'}), 400
        
        # Generate email content
        email_html = generate_weekly_email(summary_data)
        
        # ACTUALLY SEND THE EMAIL (uncommented)
        success = send_email(email, "Your Weekly Screen Time Summary", email_html)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Weekly summary sent to {email}',
                'email_sent': True
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to send email. Please check SMTP configuration.',
                'email_sent': False,
                'email_preview': email_html  # Show preview even if sending fails
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== EMAIL GENERATION ====================

def generate_weekly_email(summary_data):
    """Generate HTML email content"""
    
    date = datetime.now().strftime('%B %d, %Y')
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; text-align: center; border-radius: 10px; }}
            .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }}
            .metric {{ display: inline-block; margin: 10px 20px; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
            .tip {{ background: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; 
                   margin: 10px 0; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Your Weekly Screen Time Summary</h1>
                <p>Week ending {date}</p>
            </div>
            
            <div class="section">
                <h2>Key Metrics</h2>
                <div class="metric">
                    <div>Total Screen Time</div>
                    <div class="metric-value">{summary_data.get('total_screen_time', 'N/A')}</div>
                </div>
                <div class="metric">
                    <div>Health Status</div>
                    <div class="metric-value">{summary_data.get('health_status', 'N/A')}</div>
                </div>
            </div>
            
            <div class="tip">
                <h3>💡 This Week's Wellness Tip</h3>
                <p>Try implementing the "Digital Sunset" rule - no screens 1 hour before bed to improve sleep quality.</p>
            </div>
            
            <div class="tip">
                <h3>🎯 Weekly Goal</h3>
                <p>Reduce social media usage by 30 minutes per day this week.</p>
            </div>
            
            <p style="text-align: center; color: #999; margin-top: 40px;">
                Login to your dashboard for detailed insights and personalized recommendations.
            </p>
        </div>
    </body>
    </html>
    """
    
    return html
def send_email(to_email, subject, html_content):
    """Send email using SMTP (configure with your settings)"""
    
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "screenhealthapp@gmail.com"
    sender_password = "kmhfexorutijjimk"  # 16-char password without spaces
    
    print(f"Attempting to send email from: {sender_email}")
    print(f"SMTP Server: {smtp_server}:{smtp_port}")
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        print("Connecting to SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        print("Starting TLS...")
        server.starttls()
        print("Logging in...")
        server.login(sender_email, sender_password)
        print("Sending message...")
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email sent successfully to: {to_email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"✗ Authentication failed: {e}")
        print("Check: 1) Email and password, 2) 2FA enabled, 3) App password generated")
        return False
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False

@app.route('/api/test-email', methods=['POST'])
def test_email():
    """Test email functionality"""
    try:
        data = request.json
        test_email = data.get('email')
        
        if not test_email:
            return jsonify({'error': 'Email required'}), 400
        
        test_html = """
        <h1>Test Email from ScreenHealth AI</h1>
        <p>This is a test email to verify SMTP configuration.</p>
        <p>If you're receiving this, email setup is working correctly!</p>
        """
        
        success = send_email(test_email, "ScreenHealth AI - Test Email", test_html)
        
        return jsonify({
            'success': success,
            'message': 'Test email sent' if success else 'Failed to send test email',
            'email': test_email
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

@app.route('/api/register', methods=['POST'])
def register_user():
    """Register a new user"""
    try:
        data = request.json
        email = data.get('email')
        name = data.get('name', 'User')
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
        
        # For now, just return success (in production, save to database)
        return jsonify({
            'success': True,
            'message': f'User {name} registered successfully',
            'email': email,
            'user_id': hash(email)  # Simple ID generation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe_weekly():
    """Subscribe to weekly emails"""
    try:
        data = request.json
        email = data.get('email')
        
        if not email:
            return jsonify({'error': 'Email required'}), 400
        
        # In production, save to database
        return jsonify({
            'success': True,
            'message': f'Subscribed {email} to weekly reports',
            'next_report': (datetime.now() + timedelta(days=7)).isoformat(),
            'email': email
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# ==================== RUN APPLICATION ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SCREEN TIME MENTAL HEALTH PREDICTION API")
    print("="*70)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  - GET  /")
    print("  - GET  /api/health")
    print("  - POST /api/predict")
    print("  - POST /api/recommendations")
    print("  - POST /api/send-weekly-summary")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)