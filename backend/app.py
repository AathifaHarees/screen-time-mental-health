"""
Screen Time Mental Health Prediction System
Part 3: Flask Backend API (Production-Ready)
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
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
from database import db


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
        
        # Map frontend field names to backend field names
        field_mapping = {
            'totalScreenTime': 'Average total screen time per day',
            'socialMedia': 'Social Media (hours)',
            'entertainment': 'Entertainment (hours)',
            'workTime': 'Work/Education (hours)',
            'sleepDuration': 'Average sleep duration (hours)',
            'screenBeforeSleep': 'Use screen before sleep',
            'exercise': 'Exercise frequency',
            'addicted': 'Feel addicted to devices',
            'sleepQuality': 'Sleep quality (1–5)',
            'stress': 'Stress due to screen usage',
            'anxious': 'Anxious without device',
            'eyeStrain': 'Eye strain or headache'
        }
        
        # Apply field name mapping
        mapped_data = {}
        for frontend_key, backend_key in field_mapping.items():
            if frontend_key in user_data:
                mapped_data[backend_key] = user_data[frontend_key]
        
        # Copy any unmapped fields
        for key, value in user_data.items():
            if key not in field_mapping:
                mapped_data[key] = value
        
        # Ordinal mappings
        ordinal_mappings = {
            'Average total screen time per day': {
                'Less than 1 hour': 0, '1–2 hours': 1, '1-2 hours': 1, '3–4 hours': 2, '3-4 hours': 2,
                '5–6 hours': 3, '5-6 hours': 3, 'More than 6 hours': 4
            },
            'Social Media (hours)': {
                'Less than 1 hour': 0, '1–2 hours': 1, '1-2 hours': 1, '3–4 hours': 2, '3-4 hours': 2,
                '5–6 hours': 3, '5-6 hours': 3, 'More than 6 hours': 4
            },
            'Entertainment (hours)': {
                'Less than 1 hour': 0, '1–2 hours': 1, '1-2 hours': 1, '3–4 hours': 2, '3-4 hours': 2,
                '5–6 hours': 3, '5-6 hours': 3, 'More than 6 hours': 4
            },
            'Work/Education (hours)': {
                'Less than 1 hour': 0, '1–2 hours': 1, '1-2 hours': 1, '3–4 hours': 2, '3-4 hours': 2,
                '5–6 hours': 3, '5-6 hours': 3, 'More than 6 hours': 4
            },
            'Average sleep duration (hours)': {
                'Less than 5': 0, '5–6': 1, '5-6': 1, '6–7': 2, '6-7': 2, '7–8': 3, '7-8': 3, 'More than 8': 4
            },
            'Use screen before sleep': {
                'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4
            },
            'Exercise frequency': {
                'Never': 0, '1–2 days per week': 1, '1-2 days per week': 1, '3–4 days per week': 2, '3-4 days per week': 2, 'Daily': 3
            },
            'Feel addicted to devices': {
                'No': 0, 'Not sure': 1, 'Maybe': 2, 'Yes': 3
            }
        }
        
        processed_data = {}
        
        # Apply ordinal encoding
        for key, value in mapped_data.items():
            if key in ordinal_mappings:
                processed_data[key] = ordinal_mappings[key].get(value, 0)
            elif isinstance(value, str) and self.label_encoders and key in self.label_encoders:
                try:
                    processed_data[key] = self.label_encoders[key].transform([value])[0]
                except:
                    processed_data[key] = 0
            else:
                processed_data[key] = value
        
        return processed_data
    
    def calculate_risk_scores(self, processed_data, user_context=None):
        """Calculate various risk scores with contextual adjustments"""
        
        # Base risk calculation
        screen_time_score = (
            processed_data.get('Average total screen time per day', 0) * 5 +
            processed_data.get('Social Media (hours)', 0) * 3 +
            processed_data.get('Entertainment (hours)', 0) * 2 +
            processed_data.get('Work/Education (hours)', 0) * 1
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
        
        # CONTEXTUAL RISK ADJUSTMENT based on occupation and field
        contextual_adjustment = 0
        
        if user_context:
            occupation = user_context.get('occupation', '')
            field = user_context.get('field', '')
            work_time = processed_data.get('Work/Education (hours)', 0)
            
            # IT Professional Adjustments
            if field == 'IT':
                # Reduce work screen time penalty for IT professionals
                if work_time >= 4:  # High work screen time is expected
                    contextual_adjustment -= work_time * 0.5  # Reduce penalty
                    
                # But increase recreational screen penalty
                social_media_hours = processed_data.get('Social Media (hours)', 0)
                entertainment_hours = processed_data.get('Entertainment (hours)', 0)
                
                if social_media_hours >= 2:  # IT workers should avoid recreational screens
                    contextual_adjustment += social_media_hours * 1.5  # Increase penalty
                if entertainment_hours >= 2:
                    contextual_adjustment += entertainment_hours * 1.2
            
            # Student Adjustments
            elif occupation == 'Student':
                # High work screen time might indicate inefficient study methods
                if work_time >= 4:
                    contextual_adjustment += work_time * 0.3  # Slight penalty for inefficiency
            
            # Non-IT Professional Adjustments
            elif field == 'Non-IT' and occupation == 'Employed':
                # High work screen time is unusual and concerning
                if work_time >= 4:
                    contextual_adjustment += work_time * 0.8  # Higher penalty
        
        total_risk = screen_time_score + sleep_risk + behavioral_risk + exercise_benefit + contextual_adjustment
        
        return {
            'total_risk': max(0, total_risk),  # Ensure non-negative
            'screen_time_score': screen_time_score,
            'sleep_risk': sleep_risk,
            'behavioral_risk': behavioral_risk,
            'work_time_score': processed_data.get('Work/Education (hours)', 0),
            'contextual_adjustment': contextual_adjustment,
            'screen_index': min(100, max(0, total_risk * 1.5)),
            'sleep_health': max(0, 100 - sleep_risk * 3),
            'wellbeing_score': max(0, 100 - behavioral_risk * 2.5)
        }
    
    def predict(self, user_data):
        """Make prediction using ML model or fallback logic"""
        
        processed_data = self.preprocess_input(user_data)
        
        # Extract user context for contextual risk adjustment
        user_context = {
            'occupation': user_data.get('occupation', ''),
            'field': user_data.get('field', ''),
            'ageGroup': user_data.get('ageGroup', '')
        }
        
        risk_scores = self.calculate_risk_scores(processed_data, user_context)
        
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
    """Generate personalized recommendations based on screen time categories"""
    
    @staticmethod
    def generate_recommendations(prediction, user_data):
        """Create personalized recommendations organized by screen time categories"""
        
        risk_scores = prediction.get('risk_scores', {})
        
        # Initialize category-based recommendations
        recommendations = {
            'work_screen_recommendations': [],
            'social_media_recommendations': [],
            'entertainment_recommendations': [],
            'general_wellness_tips': [],
            'app_lock_suggestions': [],
            'meditation_exercises': [],
            'parental_controls': []
        }
        
        # ==================== WELLNESS TIPS ====================
        total_screen = user_data.get('totalScreenTime', '')
        stress_level = int(user_data.get('stress', 3))
        sleep_before_screen = user_data.get('screenBeforeSleep', '')
        age_group = user_data.get('ageGroup', '')
        addicted = user_data.get('addicted', '')
        work_time = user_data.get('workTime', '')
        occupation = user_data.get('occupation', '')
        field = user_data.get('field', '')
        social_media = user_data.get('socialMedia', '')
        entertainment = user_data.get('entertainment', '')
        
        # Personalized based on actual screen time
        if total_screen == 'More than 6 hours':
            recommendations['general_wellness_tips'].extend([
                "🚨 CRITICAL: Your 6+ hours of daily screen time is excessive and harmful",
                "⏰ URGENT: Take a 10-minute break EVERY 30 minutes",
                "👁️ MANDATORY: Practice 20-20-20 rule - Every 20 min, look 20 feet away for 20 sec",
                "🚫 Create strict device-free zones (bedroom, bathroom, dining area)",
                "📵 Set your phone to grayscale mode after 6 PM",
                "💪 Target: Reduce screen time by 2 hours this week"
            ])
        elif total_screen == '5–6 hours':
            recommendations['general_wellness_tips'].extend([
                "⚠️ Your 5-6 hours of daily screen time is above healthy limits",
                "⏰ Take 5-minute breaks every hour",
                "👁️ Practice 20-20-20 rule for eye health",
                "🚫 Establish device-free zones in your home",
                "🌅 Implement 'Digital Sunset' - no screens 1 hour before bed",
                "📊 Target: Reduce to under 4 hours daily"
            ])
        elif total_screen == '3–4 hours':
            recommendations['general_wellness_tips'].extend([
                "✅ Your screen time is moderate but can improve",
                "⏰ Continue taking hourly breaks",
                "👁️ Use blue light filters in the evening",
                "📊 Track your usage to maintain awareness",
                "🎯 Try to keep it under 4 hours"
            ])
        elif total_screen == '1–2 hours':
            recommendations['general_wellness_tips'].extend([
                "🌟 Great! Your screen time is within healthy limits",
                "✅ Keep maintaining these excellent habits",
                "📊 Continue tracking to stay accountable",
                "💡 Consider sharing your strategies with others"
            ])
        else:  # Less than 1 hour
            recommendations['general_wellness_tips'].extend([
                "🏆 Outstanding! Your screen time is exceptionally healthy",
                "✨ You're a role model for digital wellness",
                "📚 Consider mentoring others on healthy tech habits",
                "🎯 Maintain this excellent balance"
            ])
        
        # Add stress-specific tips
        if stress_level >= 4:
            recommendations['general_wellness_tips'].insert(1, f"😰 High stress level ({stress_level}/5) detected - Screen breaks are crucial for your mental health")
        
        # Add sleep-specific tips
        if sleep_before_screen in ['Always', 'Often']:
            recommendations['general_wellness_tips'].append("😴 WARNING: Using screens before sleep is severely harming your rest - Stop 1-2 hours before bed")
        
        # ==================== APP LOCK SUGGESTIONS ====================
        
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

        # ==================== OCCUPATION & FIELD-SPECIFIC RECOMMENDATIONS ====================
        
        # IT Professional Specific Recommendations
        if field == 'IT':
            if work_time in ['More than 6 hours', '5–6 hours']:
                recommendations['general_wellness_tips'].extend([
                    "💻 IT PROFESSIONAL ALERT: High work screen time is unavoidable in your field",
                    "🎯 FOCUS: Since you can't reduce work time, optimize QUALITY of screen usage",
                    "🪑 CRITICAL: Invest in ergonomic setup - monitor at eye level, proper chair",
                    "⏰ MANDATORY: Pomodoro technique - 25 min coding, 5 min break (non-negotiable)",
                    "👁️ DEVELOPER SPECIAL: Use dark themes, increase font size, reduce eye strain",
                    "🌙 CODE CURFEW: No coding 2 hours before bed - blue light disrupts sleep",
                    "💪 PROGRAMMER FITNESS: Desk exercises every hour - neck rolls, shoulder shrugs"
                ])
                
                recommendations['app_lock_suggestions'].extend([
                    "📱 IT WORKER STRATEGY: Since work screen time is high, ELIMINATE recreational screens",
                    "🚫 ZERO TOLERANCE: No social media during work breaks - use breaks for eye rest",
                    "📺 EVENING RULE: No entertainment screens after work - your eyes need recovery",
                    "🎮 GAMING BAN: If you're a developer, avoid gaming - you've had enough screen time",
                    "📱 PHONE DISCIPLINE: Use phone only for calls/messages, not browsing"
                ])
                
                recommendations['meditation_exercises'].extend([
                    "🧘 DEVELOPER MEDITATION: 5-minute guided meditation between coding sessions",
                    "👁️ EYE YOGA: Palming technique - cover eyes with palms for 30 seconds every hour",
                    "🚶 WALKING MEETINGS: Take calls while walking, not sitting at desk",
                    "🌿 NATURE BREAKS: Step outside during breaks - no indoor screen alternatives"
                ])
            
            elif work_time in ['3–4 hours']:
                recommendations['general_wellness_tips'].extend([
                    "💻 IT PROFESSIONAL: Moderate work screen time - good balance!",
                    "🎯 MAINTAIN: Keep work screen time focused and efficient",
                    "👁️ PREVENTION: Use 20-20-20 rule religiously to prevent eye strain",
                    "🪑 ERGONOMICS: Proper setup prevents long-term issues"
                ])
        
        # Non-IT Professional Recommendations
        elif field == 'Non-IT':
            if work_time in ['More than 6 hours', '5–6 hours']:
                recommendations['general_wellness_tips'].extend([
                    "💼 NON-IT PROFESSIONAL: High screen time unusual for your field",
                    "🤔 ANALYZE: Is this screen time truly necessary for your work?",
                    "📊 OPTIMIZE: Look for ways to reduce digital tasks - use phone calls instead of emails",
                    "⏰ BATCH WORK: Group screen tasks together, then take longer breaks",
                    "📝 ANALOG ALTERNATIVES: Use pen and paper when possible",
                    "🤝 FACE-TO-FACE: Replace video calls with in-person meetings when possible"
                ])
        
        # Student-Specific Recommendations
        if occupation == 'Student':
            if work_time in ['More than 6 hours', '5–6 hours']:
                recommendations['general_wellness_tips'].extend([
                    "🎓 STUDENT ALERT: High study screen time - balance is crucial",
                    "📚 STUDY SMART: Use active recall instead of passive screen reading",
                    "✍️ HANDWRITE NOTES: Better retention and less screen time",
                    "📖 PHYSICAL BOOKS: Use printed materials when possible",
                    "👥 STUDY GROUPS: In-person collaboration reduces individual screen time",
                    "⏰ TIME BLOCKING: 45 min study, 15 min screen-free break"
                ])
                
                recommendations['app_lock_suggestions'].extend([
                    "📱 STUDENT DISCIPLINE: Block social media during study hours",
                    "🎯 FOCUS APPS: Use Forest, Cold Turkey during study sessions",
                    "📺 NO NETFLIX: Entertainment screens compete with study effectiveness",
                    "🎮 GAMING LIMITS: Maximum 1 hour after completing all studies"
                ])
        
        # Employed Professional Recommendations
        elif occupation == 'Employed':
            recommendations['parental_controls'].extend([
                f"💼 WORKING PROFESSIONAL ({field}): Work-life screen balance is crucial",
                "⚖️ BOUNDARIES: Strict separation between work and personal screen time",
                "📵 AFTER-WORK DETOX: No work emails/messages after 6 PM",
                "🏠 HOME SANCTUARY: Create device-free zones at home"
            ])
        
        # Unemployed/Retired Recommendations
        elif occupation in ['Unemployed', 'Retired']:
            if total_screen in ['More than 6 hours', '5–6 hours']:
                recommendations['general_wellness_tips'].extend([
                    f"🏠 {occupation.upper()} LIFESTYLE: High screen time may indicate boredom/isolation",
                    "🎯 PURPOSE: Replace screen time with meaningful activities",
                    "🤝 SOCIAL CONNECTION: Join clubs, volunteer - real-world interaction",
                    "🌱 HOBBIES: Develop non-digital interests - gardening, crafts, sports",
                    "📚 LEARNING: Take in-person classes instead of online courses"
                ])

        # Original work time recommendations (enhanced)
        if work_time in ['More than 6 hours', '5–6 hours']:
            # Add field-specific ergonomic advice
            if field == 'IT':
                recommendations['general_wellness_tips'].extend([
                    "💻 IT ERGONOMICS: Monitor 20-26 inches away, top of screen at eye level",
                    "⌨️ KEYBOARD POSITION: Elbows at 90 degrees, wrists straight",
                    "🖱️ MOUSE TECHNIQUE: Use whole arm, not just wrist movements"
                ])
            else:
                recommendations['general_wellness_tips'].extend([
                    "💼 GENERAL ERGONOMICS: Proper chair height, feet flat on floor",
                    "📱 PHONE POSTURE: Hold phone at eye level to avoid neck strain"
                ])
        elif work_time in ['3–4 hours']:
            recommendations['general_wellness_tips'].append(f"💼 {field} PROFESSIONAL: Moderate work screen time - maintain good posture")

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
        
        # ==================== INDUSTRY-SPECIFIC RISK MITIGATION ====================
        
        # Create industry-specific risk profiles
        industry_risk_factors = {
            'IT': {
                'unavoidable_screen_time': True,
                'eye_strain_multiplier': 1.5,
                'ergonomic_priority': 'critical',
                'blue_light_exposure': 'extreme',
                'career_dependency': 'high'
            },
            'Non-IT': {
                'unavoidable_screen_time': False,
                'eye_strain_multiplier': 1.0,
                'ergonomic_priority': 'moderate',
                'blue_light_exposure': 'moderate',
                'career_dependency': 'low'
            }
        }
        
        current_industry = industry_risk_factors.get(field, industry_risk_factors['Non-IT'])
        
        # Adjust recommendations based on industry risk profile
        if current_industry['unavoidable_screen_time'] and work_time in ['More than 6 hours', '5–6 hours']:
            # For IT professionals who CANNOT reduce work screen time
            recommendations['general_wellness_tips'].insert(0, 
                "🎯 INDUSTRY REALITY: Your profession requires high screen time - we focus on OPTIMIZATION, not reduction"
            )
            
            # Add specialized IT worker recommendations
            recommendations['app_lock_suggestions'].insert(0,
                "💻 IT PROFESSIONAL STRATEGY: Since work screens are unavoidable, ZERO tolerance for recreational screens"
            )
            
            # Enhanced eye care for IT workers
            if current_industry['eye_strain_multiplier'] > 1.0:
                recommendations['meditation_exercises'].insert(0,
                    f"👁️ IT EYE CARE PROTOCOL: Your field has {current_industry['eye_strain_multiplier']}x higher eye strain risk - follow these religiously"
                )
        
        # Career-specific addiction handling
        if addicted in ['Yes', 'Maybe'] and field == 'IT':
            recommendations['parental_controls'].insert(0,
                "💻 IT ADDICTION PARADOX: You need screens for work but feel addicted - separate work and personal usage completely"
            )
        
        # ==================== WORK-LIFE INTEGRATION STRATEGIES ====================
        
        # Smart work-life balance based on occupation and field
        if occupation == 'Employed' and work_time in ['More than 6 hours', '5–6 hours']:
            work_life_strategies = []
            
            if field == 'IT':
                work_life_strategies.extend([
                    "⚖️ IT WORK-LIFE SEPARATION: Use different devices for work and personal",
                    "🖥️ DUAL SETUP: Work computer vs personal computer - never mix",
                    "📱 PHONE RULES: Work apps only during work hours",
                    "🌅 DIGITAL SUNRISE: Start work with intention, not mindless browsing",
                    "🌙 DIGITAL SUNSET: Hard cutoff - no work screens after 7 PM"
                ])
            else:
                work_life_strategies.extend([
                    "💼 PROFESSIONAL BOUNDARIES: Question if all screen time is truly necessary",
                    "📞 PHONE FIRST: Call instead of email when possible",
                    "📝 ANALOG BACKUP: Use paper for brainstorming and planning",
                    "🤝 IN-PERSON PRIORITY: Face-to-face meetings over video calls"
                ])
            
            recommendations['parental_controls'].extend(work_life_strategies)
        
        # Student-specific study optimization
        if occupation == 'Student' and work_time in ['More than 6 hours', '5–6 hours']:
            study_optimization = [
                "📚 STUDY EFFICIENCY: High screen time suggests inefficient study methods",
                "✍️ ACTIVE LEARNING: Handwrite notes, use flashcards, teach others",
                "📖 PRINT MATERIALS: Use physical textbooks when possible",
                "🎯 FOCUSED SESSIONS: 90-minute deep work blocks with 20-minute breaks",
                "👥 STUDY GROUPS: Collaborative learning reduces individual screen time"
            ]
            recommendations['general_wellness_tips'].extend(study_optimization)
        
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
        
        # Age-specific recommendations (enhanced with occupation context)
        if age_group in ['18–24']:
            if occupation == 'Student':
                recommendations['parental_controls'].extend([
                    "🎓 STUDENT (18-24): Critical time for building healthy digital habits",
                    "📚 ACADEMIC SUCCESS: Screen discipline directly impacts grades",
                    "👥 PEER ACCOUNTABILITY: Study groups with screen-time goals",
                    "💼 CAREER PREP: Develop focus skills employers value"
                ])
            else:
                recommendations['parental_controls'].extend([
                    "👥 YOUNG PROFESSIONAL: Build peer accountability with colleagues",
                    "💼 CAREER FOCUS: Replace social media time with skill-building",
                    "🎯 Your age group is most vulnerable to screen addiction - be vigilant"
                ])
        
        elif age_group in ['25–34']:
            if field == 'IT':
                recommendations['parental_controls'].extend([
                    "💻 IT PROFESSIONAL (25-34): High-earning potential requires screen discipline",
                    "🚀 CAREER GROWTH: Focus screen time on learning new technologies",
                    "⚖️ WORK-LIFE BALANCE: Prevent burnout with strict boundaries"
                ])
            else:
                recommendations['parental_controls'].extend([
                    "💼 PROFESSIONAL PRIME: Prioritize career-advancing screen usage",
                    "🤝 INDUSTRY NETWORKING: Use screens for professional development",
                    "📈 PRODUCTIVITY FOCUS: Track how screen habits affect work performance"
                ])
        
        elif age_group in ['35–44', '45–54']:
            recommendations['parental_controls'].extend([
                f"👨‍👩‍👧‍👦 PARENT & PROFESSIONAL: You're modeling behavior for your children",
                "🍽️ FAMILY RULES: Device-free meals and family time",
                "💼 LEADERSHIP: Show younger colleagues healthy screen habits",
                "📱 PARENTAL CONTROLS: Use Screen Time (iOS) or Family Link (Android)",
                "🎯 FAMILY CHALLENGES: Create screen-time reduction competitions"
            ])
            
            # Add occupation-specific parenting advice
            if field == 'IT':
                recommendations['parental_controls'].append(
                    "💻 IT PARENT SPECIAL: Teach kids that high work screen time doesn't justify recreational excess"
                )
        
        elif age_group == '55+':
            if occupation == 'Retired':
                recommendations['parental_controls'].extend([
                    "🏖️ RETIREMENT WISDOM: Use technology to enhance, not replace, real experiences",
                    "👨‍👩‍👧‍👦 GRANDPARENT ROLE: Model healthy tech use for grandchildren",
                    "🌍 EXPLORATION: Use screens to plan real-world activities, not replace them",
                    "🤝 SOCIAL CONNECTION: Video calls with family, but prioritize in-person visits"
                ])
            else:
                recommendations['parental_controls'].extend([
                    "👨‍👩‍👧‍👦 SENIOR PROFESSIONAL: Mentor younger generations on digital balance",
                    "💡 WISDOM SHARING: Your pre-digital experience is valuable - share it",
                    "⚖️ PERSPECTIVE: Balance technology benefits with traditional approaches"
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
    """Serve the main homepage"""
    try:
        # Try to serve the frontend index.html file
        return send_from_directory('../frontend', 'index.html')
    except:
        # Fallback to redirect to frontend folder
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>ScreenHealth AI</title>
            <meta http-equiv="refresh" content="0; url=/frontend/index.html">
        </head>
        <body>
            <p>Redirecting to <a href="/frontend/index.html">ScreenHealth AI</a>...</p>
        </body>
        </html>
        '''

@app.route('/api')
def api_docs():
    """API documentation"""
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

# Serve static files from frontend directory
@app.route('/frontend/<path:filename>')
def serve_frontend(filename):
    """Serve frontend static files"""
    try:
        return send_from_directory('../frontend', filename)
    except Exception as e:
        return f"File not found: {filename}", 404

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
    """Main prediction endpoint with database storage"""
    try:
        data = request.json
        user_data = data.get('user_data', {})
        user_email = data.get('email')
        
        # Validate input
        if not user_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get or create user if email provided
        user_id = None
        if user_email:
            user = db.get_user_by_email(user_email)
            if not user:
                # Create new user without password (for assessment-only users)
                result = db.create_user_with_password(
                    email=user_email,
                    name=user_data.get('name'),
                    password=None,  # No password for assessment-only users
                    age_group=user_data.get('ageGroup')
                )
                if result['success']:
                    user_id = result['user_id']
                    print(f"✓ Created new user: {user_email} (ID: {user_id})")
            else:
                user_id = user['id']
                print(f"✓ Found existing user: {user_email} (ID: {user_id})")
        
        # Make prediction
        prediction = predictor.predict(user_data)
        
        # Generate recommendations
        recommendations = RecommendationEngine.generate_recommendations(prediction, user_data)
        
        # Save to database if user exists
        if user_id:
            print(f"💾 Saving assessment for user ID: {user_id}")
            save_result = db.save_assessment(
                user_id=user_id,
                assessment_data=user_data,
                prediction_result=prediction,
                risk_scores=prediction['risk_scores']
            )
            if save_result['success']:
                print(f"✅ Assessment saved successfully! Assessment ID: {save_result['assessment_id']}")
            else:
                print(f"❌ Failed to save assessment: {save_result['error']}")
        else:
            print("⚠️ No user ID - assessment not saved to database")
        
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
            'user_id': user_id,
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

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate downloadable HTML report"""
    try:
        data = request.json
        user_data = data.get('user_data', {})
        prediction = data.get('prediction', {})
        recommendations = data.get('recommendations', {})
        user_email = data.get('email', 'user@example.com')
        
        # Get user stats if email provided
        user_stats = {}
        if user_email != 'user@example.com':
            user = db.get_user_by_email(user_email)
            if user:
                user_stats = db.get_user_stats(user['id'])
        
        # Generate comprehensive HTML report
        report_html = generate_report_html(user_data, prediction, recommendations, user_stats, user_email)
        
        return jsonify({
            'success': True,
            'report_html': report_html,
            'filename': f"screenhealth_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_report_html(user_data, prediction, recommendations, user_stats, user_email):
    """Generate comprehensive HTML report"""
    
    date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    
    # Get risk scores
    risk_scores = prediction.get('risk_scores', {})
    total_risk = risk_scores.get('total_risk', 0)
    screen_index = risk_scores.get('screen_index', 0)
    sleep_health = risk_scores.get('sleep_health', 0)
    wellbeing_score = risk_scores.get('wellbeing_score', 0)
    
    # Health status
    is_healthy = prediction.get('is_healthy', False)
    confidence = prediction.get('confidence', 0)
    status = 'Healthy' if is_healthy else 'Needs Improvement'
    status_color = '#28a745' if is_healthy else '#dc3545'
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ScreenHealth AI Report - {date}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: #f8f9fa;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin: 20px 0;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .status-badge {{
                display: inline-block;
                padding: 10px 20px;
                border-radius: 25px;
                color: white;
                font-weight: bold;
                font-size: 1.1em;
                background: {status_color};
            }}
            .metric {{
                display: inline-block;
                margin: 15px 20px;
                text-align: center;
                min-width: 120px;
            }}
            .metric-value {{
                font-size: 2.5em;
                font-weight: bold;
                color: #667eea;
                display: block;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .recommendation {{
                background: #e3f2fd;
                padding: 15px;
                border-left: 4px solid #2196f3;
                margin: 10px 0;
                border-radius: 4px;
            }}
            .recommendation h4 {{
                margin: 0 0 10px 0;
                color: #1976d2;
            }}
            .recommendation ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            .recommendation li {{
                margin: 5px 0;
            }}
            .warning {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                color: #856404;
            }}
            .danger {{
                background: #f8d7da;
                border-left: 4px solid #dc3545;
                color: #721c24;
            }}
            .success {{
                background: #d4edda;
                border-left: 4px solid #28a745;
                color: #155724;
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 40px;
                padding: 20px;
                border-top: 1px solid #ddd;
            }}
            .progress-bar {{
                width: 100%;
                height: 20px;
                background: #e9ecef;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
                transition: width 0.3s ease;
            }}
            @media print {{
                body {{ background: white; }}
                .section {{ box-shadow: none; border: 1px solid #ddd; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 ScreenHealth AI Report</h1>
            <p>Digital Wellness Assessment</p>
            <p>{date}</p>
            <p>User: {user_email}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="recommendation {'success' if is_healthy else 'danger'}">
                <h4>📋 Assessment Overview</h4>
                <p><strong>Health Status:</strong> {status} (Confidence: {confidence:.1%})</p>
                <p><strong>Risk Level:</strong> {
                    'Low Risk' if total_risk < 30 else 
                    'Moderate Risk' if total_risk < 60 else 
                    'High Risk'
                } (Score: {total_risk:.1f}/100)</p>
                <p><strong>Key Findings:</strong></p>
                <ul>
                    <li>Total daily screen time: {user_data.get('totalScreenTime', 'Not specified')}</li>
                    <li>Social media usage: {user_data.get('socialMedia', 'Not specified')}</li>
                    <li>Sleep quality: {user_data.get('sleepQuality', 'Not specified')}/5</li>
                    <li>Stress level: {user_data.get('stress', 'Not specified')}/5</li>
                    <li>Exercise frequency: {user_data.get('exercise', 'Not specified')}</li>
                </ul>
                
                <p><strong>Summary:</strong> 
                {'Your screen time habits appear to be within healthy limits. Continue maintaining these positive patterns while staying mindful of your digital wellness.' if is_healthy else 
                'Your screen time patterns indicate areas for improvement. The recommendations below will help you develop healthier digital habits and reduce potential negative impacts on your mental health.'}
                </p>
                
                <p><strong>Priority Actions:</strong></p>
                <ul>
                    {f'<li>🚨 Reduce total screen time from {user_data.get("totalScreenTime", "current level")} - this is your highest priority</li>' if user_data.get('totalScreenTime') in ['More than 6 hours', '5–6 hours'] else ''}
                    {f'<li>😴 Improve sleep quality (currently {user_data.get("sleepQuality", "N/A")}/5) by avoiding screens before bed</li>' if int(user_data.get('sleepQuality', 5)) <= 3 else ''}
                    {f'<li>🧘 Manage stress levels (currently {user_data.get("stress", "N/A")}/5) with regular breaks and mindfulness</li>' if int(user_data.get('stress', 1)) >= 4 else ''}
                    {f'<li>💪 Increase physical activity from "{user_data.get("exercise", "current level")}" to daily exercise</li>' if user_data.get('exercise') in ['Never', '1–2 days per week'] else ''}
                    <li>📊 Take regular assessments to track your progress over time</li>
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>Overall Health Status</h2>
            <div style="text-align: center; margin: 20px 0;">
                <div class="status-badge">{status}</div>
                <p style="margin-top: 15px;">Confidence: {confidence:.1%}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Risk Assessment Scores</h2>
            <div style="text-align: center;">
                <div class="metric">
                    <span class="metric-value">{total_risk:.1f}</span>
                    <span class="metric-label">Total Risk</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{screen_index:.1f}</span>
                    <span class="metric-label">Screen Index</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{sleep_health:.1f}</span>
                    <span class="metric-label">Sleep Health</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{wellbeing_score:.1f}</span>
                    <span class="metric-label">Wellbeing</span>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <p><strong>Risk Level Progress:</strong></p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(100, total_risk)}%;"></div>
                </div>
                <p style="font-size: 0.9em; color: #666;">
                    0-30: Low Risk | 30-60: Moderate Risk | 60+: High Risk
                </p>
            </div>
        </div>
    """
    
    # Add user statistics if available
    if user_stats:
        html += f"""
        <div class="section">
            <h2>Your Progress Statistics</h2>
            <div style="text-align: center;">
                <div class="metric">
                    <span class="metric-value">{user_stats.get('total_assessments', 0)}</span>
                    <span class="metric-label">Total Assessments</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{user_stats.get('avg_risk_score', 0):.1f}</span>
                    <span class="metric-label">Average Risk</span>
                </div>
                <div class="metric">
                    <span class="metric-value">{user_stats.get('healthy_count', 0)}</span>
                    <span class="metric-label">Healthy Results</span>
                </div>
            </div>
            
            <div class="recommendation">
                <h4>📈 Progress Analysis</h4>
                <p>You have completed {user_stats.get('total_assessments', 0)} assessment(s) with an average risk score of {user_stats.get('avg_risk_score', 0):.1f}. 
                Out of these, {user_stats.get('healthy_count', 0)} showed healthy patterns while {user_stats.get('unhealthy_count', 0)} indicated areas for improvement.</p>
                
                {'<p><strong>Positive Trend:</strong> You are consistently maintaining healthy digital habits. Keep up the excellent work!</p>' if user_stats.get('healthy_count', 0) > user_stats.get('unhealthy_count', 0) else 
                '<p><strong>Improvement Needed:</strong> Your assessments show a pattern that needs attention. Focus on the recommendations below to develop healthier habits.</p>' if user_stats.get('unhealthy_count', 0) > user_stats.get('healthy_count', 0) else 
                '<p><strong>Mixed Results:</strong> Your assessments show both healthy and concerning patterns. Consistency in following recommendations will help improve your overall digital wellness.</p>'}
            </div>
        </div>
        """
    
    # Add detailed text analysis
    html += f"""
        <div class="section">
            <h2>Detailed Analysis</h2>
            <div class="recommendation">
                <h4>🔍 Screen Time Analysis</h4>
                <p><strong>Total Screen Time:</strong> {user_data.get('totalScreenTime', 'Not specified')}</p>
                <p>{'Your total screen time is within healthy limits. This is excellent for your mental health and overall wellbeing.' if user_data.get('totalScreenTime') in ['Less than 1 hour', '1–2 hours'] else
                'Your total screen time is moderate. Consider implementing regular breaks and mindful usage to maintain balance.' if user_data.get('totalScreenTime') == '3–4 hours' else
                'Your total screen time is above recommended levels. This may be impacting your sleep, stress levels, and overall mental health. Immediate action is recommended.' if user_data.get('totalScreenTime') in ['5–6 hours', 'More than 6 hours'] else
                'Screen time data not available for analysis.'}</p>
                
                <p><strong>Social Media Usage:</strong> {user_data.get('socialMedia', 'Not specified')}</p>
                <p>{'Excellent social media discipline! This low usage supports better mental health and real-world connections.' if user_data.get('socialMedia') in ['Less than 1 hour', '1–2 hours'] else
                'Moderate social media usage. Be mindful of the content you consume and take regular breaks.' if user_data.get('socialMedia') == '3–4 hours' else
                'High social media usage detected. This can significantly impact mental health, sleep, and productivity. Consider setting strict daily limits.' if user_data.get('socialMedia') in ['5–6 hours', 'More than 6 hours'] else
                'Social media usage data not available.'}</p>
            </div>
            
            <div class="recommendation">
                <h4>😴 Sleep & Wellness Analysis</h4>
                <p><strong>Sleep Quality:</strong> {user_data.get('sleepQuality', 'Not specified')}/5</p>
                <p>{'Excellent sleep quality! This is crucial for mental health and cognitive function.' if int(user_data.get('sleepQuality', 0)) >= 4 else
                'Moderate sleep quality. Consider improving sleep hygiene and reducing screen time before bed.' if int(user_data.get('sleepQuality', 0)) == 3 else
                'Poor sleep quality detected. This is likely connected to your screen time habits and needs immediate attention.' if int(user_data.get('sleepQuality', 0)) <= 2 else
                'Sleep quality data not available.'}</p>
                
                <p><strong>Stress Level:</strong> {user_data.get('stress', 'Not specified')}/5</p>
                <p>{'Low stress levels indicate good mental health balance.' if int(user_data.get('stress', 0)) <= 2 else
                'Moderate stress levels. Screen time management and regular breaks can help reduce stress.' if int(user_data.get('stress', 0)) == 3 else
                'High stress levels detected. Excessive screen time may be contributing to this. Immediate stress management techniques are recommended.' if int(user_data.get('stress', 0)) >= 4 else
                'Stress level data not available.'}</p>
                
                <p><strong>Exercise Frequency:</strong> {user_data.get('exercise', 'Not specified')}</p>
                <p>{'Excellent exercise routine! Regular physical activity is crucial for counteracting screen time effects.' if user_data.get('exercise') == 'Daily' else
                'Good exercise frequency. This helps balance screen time effects on your physical and mental health.' if user_data.get('exercise') == '3–4 days per week' else
                'Limited exercise detected. Increasing physical activity will significantly improve your overall wellbeing and help manage screen time effects.' if user_data.get('exercise') in ['1–2 days per week', 'Never'] else
                'Exercise data not available.'}</p>
            </div>
            
            <div class="recommendation">
                <h4>🎯 Risk Assessment Summary</h4>
                <p><strong>Overall Risk Score:</strong> {total_risk:.1f}/100</p>
                <p>{'Your risk score indicates healthy digital habits. Continue monitoring and maintaining these positive patterns.' if total_risk < 30 else
                'Your risk score suggests moderate concern. Implementing the recommendations below will help improve your digital wellness.' if total_risk < 60 else
                'Your risk score indicates high concern. Immediate action is needed to prevent negative impacts on your mental health and wellbeing.'}</p>
                
                <p><strong>Key Risk Factors:</strong></p>
                <ul>
                    {f'<li>Excessive total screen time ({user_data.get("totalScreenTime", "N/A")})</li>' if user_data.get('totalScreenTime') in ['More than 6 hours', '5–6 hours'] else ''}
                    {f'<li>High social media usage ({user_data.get("socialMedia", "N/A")})</li>' if user_data.get('socialMedia') in ['More than 6 hours', '5–6 hours'] else ''}
                    {f'<li>Poor sleep quality ({user_data.get("sleepQuality", "N/A")}/5)</li>' if int(user_data.get('sleepQuality', 5)) <= 2 else ''}
                    {f'<li>High stress levels ({user_data.get("stress", "N/A")}/5)</li>' if int(user_data.get('stress', 1)) >= 4 else ''}
                    {f'<li>Insufficient physical activity ({user_data.get("exercise", "N/A")})</li>' if user_data.get('exercise') in ['Never', '1–2 days per week'] else ''}
                    {f'<li>Potential device addiction (feels addicted: {user_data.get("addicted", "N/A")})</li>' if user_data.get('addicted') in ['Yes', 'Maybe'] else ''}
                </ul>
                
                {f'<p><strong>Protective Factors:</strong></p><ul>' if any([
                    user_data.get('totalScreenTime') in ['Less than 1 hour', '1–2 hours'],
                    user_data.get('socialMedia') in ['Less than 1 hour', '1–2 hours'],
                    int(user_data.get('sleepQuality', 0)) >= 4,
                    int(user_data.get('stress', 5)) <= 2,
                    user_data.get('exercise') in ['Daily', '3–4 days per week']
                ]) else ''}
                {f'<li>Healthy total screen time ({user_data.get("totalScreenTime", "N/A")})</li>' if user_data.get('totalScreenTime') in ['Less than 1 hour', '1–2 hours'] else ''}
                {f'<li>Controlled social media usage ({user_data.get("socialMedia", "N/A")})</li>' if user_data.get('socialMedia') in ['Less than 1 hour', '1–2 hours'] else ''}
                {f'<li>Good sleep quality ({user_data.get("sleepQuality", "N/A")}/5)</li>' if int(user_data.get('sleepQuality', 0)) >= 4 else ''}
                {f'<li>Low stress levels ({user_data.get("stress", "N/A")}/5)</li>' if int(user_data.get('stress', 5)) <= 2 else ''}
                {f'<li>Regular exercise routine ({user_data.get("exercise", "N/A")})</li>' if user_data.get('exercise') in ['Daily', '3–4 days per week'] else ''}
                {'</ul>' if any([
                    user_data.get('totalScreenTime') in ['Less than 1 hour', '1–2 hours'],
                    user_data.get('socialMedia') in ['Less than 1 hour', '1–2 hours'],
                    int(user_data.get('sleepQuality', 0)) >= 4,
                    int(user_data.get('stress', 5)) <= 2,
                    user_data.get('exercise') in ['Daily', '3–4 days per week']
                ]) else ''}
            </div>
        </div>
    """
    
    # Add recommendations
    if recommendations:
        html += """
        <div class="section">
            <h2>Personalized Recommendations</h2>
        """
        
        # Wellness tips
        if recommendations.get('wellness_tips'):
            html += """
            <div class="recommendation success">
                <h4>🌟 Wellness Tips</h4>
                <ul>
            """
            for tip in recommendations['general_wellness_tips'][:5]:  # Limit to 5 tips
                html += f"<li>{tip}</li>"
            html += "</ul></div>"
        
        # App lock suggestions
        if recommendations.get('app_lock_suggestions'):
            html += """
            <div class="recommendation warning">
                <h4>📱 Screen Time Management</h4>
                <ul>
            """
            for suggestion in recommendations['app_lock_suggestions'][:5]:
                html += f"<li>{suggestion}</li>"
            html += "</ul></div>"
        
        # Meditation exercises
        if recommendations.get('meditation_exercises'):
            html += """
            <div class="recommendation">
                <h4>🧘 Mindfulness & Exercise</h4>
                <ul>
            """
            for exercise in recommendations['meditation_exercises'][:5]:
                html += f"<li>{exercise}</li>"
            html += "</ul></div>"
        
        # Parental controls / accountability
        if recommendations.get('parental_controls'):
            html += """
            <div class="recommendation danger">
                <h4>🎯 Accountability & Support</h4>
                <ul>
            """
            for control in recommendations['parental_controls'][:5]:
                html += f"<li>{control}</li>"
            html += "</ul></div>"
        
        html += "</div>"
    
    # Add assessment data summary
    html += f"""
        <div class="section">
            <h2>Assessment Summary</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4>Screen Time Usage</h4>
                    <p><strong>Total Screen Time:</strong> {user_data.get('totalScreenTime', 'N/A')}</p>
                    <p><strong>Social Media:</strong> {user_data.get('socialMedia', 'N/A')}</p>
                    <p><strong>Entertainment:</strong> {user_data.get('entertainment', 'N/A')}</p>
                    <p><strong>Work Time:</strong> {user_data.get('workTime', 'N/A')}</p>
                </div>
                <div>
                    <h4>Health Indicators</h4>
                    <p><strong>Sleep Quality:</strong> {user_data.get('sleepQuality', 'N/A')}/5</p>
                    <p><strong>Stress Level:</strong> {user_data.get('stress', 'N/A')}/5</p>
                    <p><strong>Exercise Frequency:</strong> {user_data.get('exercise', 'N/A')}</p>
                    <p><strong>Feel Addicted:</strong> {user_data.get('addicted', 'N/A')}</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>ScreenHealth AI</strong> - Digital Wellness Assessment</p>
            <p>Generated on {date}</p>
            <p style="font-size: 0.9em; color: #999;">
                This report is for educational purposes. For serious health concerns, consult healthcare professionals.
            </p>
        </div>
    </body>
    </html>
    """
    
    return html

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
    """Send email using SMTP - direct configuration"""
    
    # Direct configuration from .env
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "screenhealthapp@gmail.com"
    sender_password = "qesujuzgmhheuwth"
    
    print(f"Sending email from: {sender_email} to: {to_email}")
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✓ Email sent successfully!")
        return True
    except Exception as e:
        print(f"✗ Email failed: {e}")
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

@app.route('/api/login', methods=['POST'])
def login_user():
    """User login endpoint"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Get user from database
        user = db.get_user_by_email(email)
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # For now, we'll use a simple password check
        # In production, use proper password hashing (bcrypt, etc.)
        stored_password = user.get('password', '')
        if password != stored_password:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Log the login activity
        db.log_email(
            user_id=user['id'],
            email_type='login',
            subject='User Login',
            success=True
        )
        
        return jsonify({
            'success': True,
            'user_id': user['id'],
            'email': user['email'],
            'name': user['name'],
            'message': 'Login successful'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    """Enhanced user registration with password"""
    try:
        data = request.json
        email = data.get('email')
        name = data.get('name', 'User')
        password = data.get('password')
        age_group = data.get('age_group')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Create user in database with password
        result = db.create_user_with_password(email=email, name=name, password=password, age_group=age_group)
        
        if result['success']:
            # Log the registration
            db.log_email(
                user_id=result['user_id'],
                email_type='registration',
                subject='User Registration',
                success=True
            )
            
            return jsonify({
                'success': True,
                'message': result['message'],
                'user_id': result['user_id'],
                'email': email
            })
        else:
            return jsonify({'error': result['error']}), 400
    
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
        
        # Get user and enable weekly emails
        user = db.get_user_by_email(email)
        if not user:
            return jsonify({'error': 'User not found. Please register first.'}), 404
        
        db.enable_weekly_emails(user['id'], True)
        
        return jsonify({
            'success': True,
            'message': f'Subscribed {email} to weekly reports',
            'next_report': (datetime.now() + timedelta(days=7)).isoformat(),
            'email': email
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# New database-related endpoints
@app.route('/api/user/<email>/history', methods=['GET'])
def get_user_history(email):
    """Get user's assessment history"""
    try:
        user = db.get_user_by_email(email)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        limit = request.args.get('limit', 10, type=int)
        assessments = db.get_user_assessments(user['id'], limit)
        
        return jsonify({
            'user': dict(user),
            'assessments': assessments,
            'total_count': len(assessments)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<email>/stats', methods=['GET'])
def get_user_stats(email):
    """Get user statistics"""
    try:
        user = db.get_user_by_email(email)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        stats = db.get_user_stats(user['id'])
        
        return jsonify({
            'user': dict(user),
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    """Get overall system statistics"""
    try:
        stats = db.get_database_stats()
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/emails', methods=['GET'])
def get_email_logs():
    """Get email sending logs for tracking"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get recent email logs with user info
        cursor.execute('''
            SELECT 
                e.id, e.email_type, e.subject, e.sent_at, e.success, e.error_message,
                u.email, u.name
            FROM email_logs e
            JOIN users u ON e.user_id = u.id
            ORDER BY e.sent_at DESC
            LIMIT 50
        ''')
        
        logs = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'email_logs': [dict(log) for log in logs],
            'total_count': len(logs)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/emails', methods=['GET'])
def get_all_user_emails():
    """Get all user emails for tracking purposes"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, email, name, age_group, created_at, last_assessment, 
                weekly_email_enabled,
                (SELECT COUNT(*) FROM assessments WHERE user_id = users.id) as assessment_count
            FROM users 
            ORDER BY created_at DESC
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return jsonify({
            'users': [dict(user) for user in users],
            'total_users': len(users),
            'email_subscribers': len([u for u in users if u['weekly_email_enabled']])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<email>/profile', methods=['PUT'])
def update_user_profile(email):
    """Update user profile"""
    try:
        data = request.get_json()
        
        # Get user by email
        user = db.get_user_by_email(email)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Extract update fields
        name = data.get('name')
        age_group = data.get('age_group')
        password = data.get('password')
        
        # Validate inputs
        if name and len(name.strip()) < 2:
            return jsonify({'error': 'Name must be at least 2 characters'}), 400
        
        if password and len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Update profile
        result = db.update_user_profile(
            user_id=user['id'],
            name=name.strip() if name else None,
            age_group=age_group if age_group else None,
            password=password if password else None
        )
        
        if result['success']:
            # Return updated user info
            updated_user = db.get_user_by_email(email)
            return jsonify({
                'success': True,
                'message': result['message'],
                'user': {
                    'id': updated_user['id'],
                    'email': updated_user['email'],
                    'name': updated_user['name'],
                    'age_group': updated_user['age_group']
                }
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        print(f"Profile update error: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== RUN APPLICATION ====================

if __name__ == '__main__':
    import os
    
    print("\n" + "="*70)
    print("SCREEN TIME MENTAL HEALTH PREDICTION API")
    print("="*70)
    print("\nStarting Flask server...")
    
    # Get port from environment variable (for deployment) or use 5000 for local
    port = int(os.environ.get('PORT', 5000))
    
    print(f"API will be available at: http://localhost:{port}")
    print("\nEndpoints:")
    print("  - GET  /")
    print("  - GET  /api/health")
    print("  - POST /api/predict")
    print("  - POST /api/generate-report")
    print("  - POST /api/register")
    print("  - POST /api/login")
    print("  - GET  /api/user/<email>/history")
    print("  - GET  /api/user/<email>/stats")
    print("\n" + "="*70 + "\n")
    
    # Production vs Development
    is_production = os.environ.get('FLASK_ENV') == 'production'
    app.run(debug=not is_production, host='0.0.0.0', port=port)
