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
            'general_wellness_tips': []
        }
        
        # Extract user data
        occupation = user_data.get('occupation', '')
        field = user_data.get('field', '')
        age_group = user_data.get('ageGroup', '')
        work_time = user_data.get('workTime', '')
        social_media = user_data.get('socialMedia', '')
        entertainment = user_data.get('entertainment', '')
        total_screen = user_data.get('totalScreenTime', '')
        stress_level = int(user_data.get('stress', 3))
        sleep_before_screen = user_data.get('screenBeforeSleep', '')
        addicted = user_data.get('addicted', '')
        
        # ==================== WORK SCREEN TIME RECOMMENDATIONS ====================
        
        if work_time in ['More than 6 hours', '5–6 hours']:
            if field == 'IT':
                recommendations['work_screen_recommendations'].extend([
                    "💻 IT PROFESSIONAL REALITY: Your high work screen time is unavoidable - focus on OPTIMIZATION",
                    "🎯 ERGONOMIC SETUP: Monitor 20-26 inches away, top of screen at eye level",
                    "⌨️ PROPER POSITIONING: Elbows at 90 degrees, wrists straight while typing",
                    "⏰ POMODORO TECHNIQUE: 25 minutes coding, 5 minutes break (non-negotiable)",
                    "👁️ DEVELOPER EYE CARE: Use dark themes, increase font size, reduce blue light",
                    "🌙 CODE CURFEW: No work screens 2 hours before bed - blue light disrupts sleep",
                    "💪 DESK EXERCISES: Neck rolls, shoulder shrugs every hour",
                    "🖱️ MOUSE TECHNIQUE: Use whole arm movement, not just wrist"
                ])
            elif field == 'Non-IT':
                recommendations['work_screen_recommendations'].extend([
                    "💼 UNUSUAL PATTERN: High work screen time is uncommon for your field",
                    "🤔 NECESSITY CHECK: Question if all this screen time is truly required",
                    "📞 PHONE FIRST: Replace emails with phone calls when possible",
                    "📝 ANALOG ALTERNATIVES: Use pen and paper for brainstorming and notes",
                    "🤝 IN-PERSON MEETINGS: Choose face-to-face over video calls",
                    "⏰ BATCH PROCESSING: Group all screen tasks together, then take longer breaks",
                    "🪑 ERGONOMICS: Proper chair height, feet flat on floor"
                ])
            
            if occupation == 'Student':
                recommendations['work_screen_recommendations'].extend([
                    "🎓 STUDY EFFICIENCY: High study screen time suggests inefficient methods",
                    "✍️ ACTIVE LEARNING: Handwrite notes for better retention and less screen time",
                    "📚 PHYSICAL MATERIALS: Use printed textbooks and materials when possible",
                    "🎯 FOCUSED SESSIONS: 90-minute deep work blocks with 20-minute screen-free breaks",
                    "👥 STUDY GROUPS: Collaborative learning reduces individual screen time",
                    "📖 READING TECHNIQUE: Use physical books for leisure reading"
                ])
        
        elif work_time in ['3–4 hours']:
            recommendations['work_screen_recommendations'].extend([
                f"💼 MODERATE WORK USAGE ({field}): Good balance for your profession",
                "🎯 MAINTAIN EFFICIENCY: Keep work screen time focused and productive",
                "👁️ PREVENTION: Use 20-20-20 rule to prevent eye strain buildup",
                "🪑 POSTURE CHECK: Maintain good ergonomics even for moderate usage"
            ])
        
        elif work_time in ['1–2 hours', 'Less than 1 hour']:
            recommendations['work_screen_recommendations'].extend([
                f"✅ EXCELLENT WORK BALANCE ({field}): Your work screen time is healthy",
                "🎯 MAINTAIN: Keep work screen usage efficient and purposeful",
                "💡 ROLE MODEL: Share your strategies with colleagues who struggle"
            ])
        
        # ==================== SOCIAL MEDIA RECOMMENDATIONS ====================
        
        if social_media == 'More than 6 hours':
            recommendations['social_media_recommendations'].extend([
                "🚨 CRITICAL ALERT: 6+ hours on social media is severely damaging your mental health",
                "🔒 IMMEDIATE ACTION: Set 1-hour daily limit on ALL social apps RIGHT NOW",
                "📱 APP REMOVAL: DELETE Instagram, TikTok, Facebook for a 1-week digital detox",
                "⏱️ STRICT SCHEDULE: Allow access ONLY during 12-1 PM, no exceptions",
                "⚫ GRAYSCALE MODE: Force your phone to grayscale permanently",
                "🔔 NOTIFICATION BAN: Disable ALL social media notifications immediately",
                "🧠 DOPAMINE RESET: Replace social scrolling with real-world activities"
            ])
            
            if field == 'IT':
                recommendations['social_media_recommendations'].insert(1,
                    "💻 IT WORKER CRITICAL: Your work already requires high screen time - ZERO tolerance for social media"
                )
        
        elif social_media == '5–6 hours':
            recommendations['social_media_recommendations'].extend([
                "📱 HIGH RISK: 5-6 hours on social media is excessive and harmful",
                "⏱️ URGENT LIMIT: Set strict 2-hour daily limit using app timers",
                "🔒 APP CONTROLS: Instagram (30min), TikTok (30min), Twitter (30min) maximum",
                "⚫ EVENING GRAYSCALE: Switch to grayscale after 8 PM",
                "🔔 NOTIFICATION CLEANUP: Disable all non-essential notifications",
                "📅 TIME WINDOWS: Limit access to 12-1 PM and 6-7 PM only"
            ])
        
        elif social_media == '3–4 hours':
            recommendations['social_media_recommendations'].extend([
                "📱 ABOVE HEALTHY LIMITS: 3-4 hours on social media needs reduction",
                "⏱️ TARGET REDUCTION: Aim for 2 hours daily using built-in app timers",
                "🔔 NOTIFICATION CONTROL: Turn off all non-essential alerts",
                "⚫ NIGHT MODE: Use grayscale mode after 9 PM",
                "🎯 WEEKLY GOAL: Reduce by 30 minutes this week"
            ])
        
        elif social_media == '1–2 hours':
            recommendations['social_media_recommendations'].extend([
                "✅ REASONABLE USAGE: Your 1-2 hours of social media is acceptable",
                "📱 MAINTAIN LIMITS: Keep current healthy boundaries",
                "🎯 OPTIMIZATION: Consider reducing to under 1 hour for even better wellbeing",
                "🔔 MINIMAL NOTIFICATIONS: Keep only essential alerts enabled"
            ])
        
        else:  # Less than 1 hour
            recommendations['social_media_recommendations'].extend([
                "🌟 EXEMPLARY CONTROL: Your social media usage is outstanding",
                "✅ DIGITAL DISCIPLINE: You've mastered healthy social media habits",
                "💡 MENTOR OTHERS: Share your strategies with friends who struggle",
                "🎯 MAINTAIN EXCELLENCE: Keep this healthy boundary"
            ])
        
        # ==================== ENTERTAINMENT RECOMMENDATIONS ====================
        
        if entertainment == 'More than 6 hours':
            recommendations['entertainment_recommendations'].extend([
                "🎮 EXCESSIVE ENTERTAINMENT: 6+ hours is severely impacting your life balance",
                "⏱️ EMERGENCY LIMIT: Set strict 2-hour daily maximum immediately",
                "📺 STREAMING CONTROL: Use Netflix/YouTube time limits and parental controls",
                "🎯 REPLACEMENT ACTIVITIES: Find real-world hobbies to replace screen entertainment",
                "⚫ DEVICE REMOVAL: Remove entertainment apps from bedroom and dining areas",
                "🕐 SCHEDULED VIEWING: Only 7-9 PM for entertainment screens"
            ])
        
        elif entertainment == '5–6 hours':
            recommendations['entertainment_recommendations'].extend([
                "🎬 HIGH ENTERTAINMENT USAGE: 5-6 hours is above healthy limits",
                "⏱️ REDUCTION TARGET: Aim for 3 hours daily maximum",
                "📺 MINDFUL VIEWING: Choose quality content over mindless scrolling",
                "🎯 ACTIVE ALTERNATIVES: Replace 2 hours with physical activities",
                "⚫ EVENING LIMITS: No entertainment screens after 9 PM"
            ])
        
        elif entertainment == '3–4 hours':
            recommendations['entertainment_recommendations'].extend([
                "🎬 MODERATE ENTERTAINMENT: 3-4 hours can be optimized",
                "⏱️ QUALITY FOCUS: Choose intentional viewing over random browsing",
                "🎯 BALANCE GOAL: Try to reduce to 2-3 hours daily",
                "📺 SCHEDULED VIEWING: Set specific times for entertainment"
            ])
        
        elif entertainment == '1–2 hours':
            recommendations['entertainment_recommendations'].extend([
                "✅ HEALTHY ENTERTAINMENT: Your 1-2 hours is well-balanced",
                "🎬 QUALITY CONTENT: Continue choosing meaningful entertainment",
                "🎯 MAINTAIN BALANCE: Keep this healthy entertainment routine"
            ])
        
        else:  # Less than 1 hour
            recommendations['entertainment_recommendations'].extend([
                "🌟 EXCELLENT BALANCE: Your entertainment screen time is optimal",
                "✅ LIFE BALANCE: You prioritize real-world activities perfectly",
                "💡 INSPIRATION: You're a role model for balanced living"
            ])
        
        # ==================== GENERAL WELLNESS TIPS ====================
        
        # Overall screen time assessment
        if total_screen == 'More than 6 hours':
            recommendations['general_wellness_tips'].extend([
                "🚨 TOTAL SCREEN TIME CRITICAL: 6+ hours daily is severely harmful",
                "⏰ MANDATORY BREAKS: 10-minute break EVERY 30 minutes of screen use",
                "👁️ EYE PROTECTION: 20-20-20 rule is non-negotiable for your usage level",
                "🚫 DEVICE-FREE ZONES: Bedroom, bathroom, and dining area must be screen-free",
                "📵 DIGITAL SUNSET: All screens off 1-2 hours before bedtime"
            ])
        elif total_screen == '5–6 hours':
            recommendations['general_wellness_tips'].extend([
                "⚠️ HIGH TOTAL USAGE: 5-6 hours daily is above healthy limits",
                "⏰ REGULAR BREAKS: 5-minute break every hour of screen use",
                "👁️ EYE CARE: Practice 20-20-20 rule consistently",
                "🌅 DIGITAL BOUNDARIES: Implement 'Digital Sunset' routine"
            ])
        
        # Stress and sleep integration
        if stress_level >= 4:
            recommendations['general_wellness_tips'].insert(0,
                f"😰 HIGH STRESS ALERT: Your stress level ({stress_level}/5) requires immediate screen time reduction"
            )
        
        if sleep_before_screen in ['Always', 'Often']:
            recommendations['general_wellness_tips'].append(
                "😴 SLEEP DISRUPTION: Using screens before bed is severely harming your rest quality"
            )
        
        # Addiction awareness
        if addicted == 'Yes':
            recommendations['general_wellness_tips'].insert(0,
                "🚨 ADDICTION ACKNOWLEDGED: You recognize device addiction - seek accountability partner immediately"
            )
        
        # Age and occupation specific general advice
        if age_group in ['18–24'] and occupation == 'Student':
            recommendations['general_wellness_tips'].append(
                "🎓 STUDENT SUCCESS: Your screen habits directly impact academic performance"
            )
        elif field == 'IT' and work_time in ['More than 6 hours', '5–6 hours']:
            recommendations['general_wellness_tips'].append(
                "💻 IT PROFESSIONAL: Since work screens are unavoidable, eliminate ALL recreational screen time"
            )
        
        return recommendations