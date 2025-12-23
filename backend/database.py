"""
Database module for ScreenHealth AI
SQLite database operations for user data and assessments
"""

import sqlite3
import json
from datetime import datetime
import os

class ScreenHealthDB:
    def __init__(self, db_path='screenhealth.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT,
                name TEXT,
                age_group TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_assessment TIMESTAMP,
                last_login TIMESTAMP,
                weekly_email_enabled BOOLEAN DEFAULT 0,
                total_logins INTEGER DEFAULT 0
            )
        ''')
        
        # Assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                assessment_data TEXT NOT NULL,  -- JSON string
                prediction_result TEXT NOT NULL,  -- JSON string
                risk_scores TEXT NOT NULL,  -- JSON string
                is_healthy BOOLEAN,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Weekly summaries table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weekly_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                week_start DATE,
                week_end DATE,
                total_assessments INTEGER,
                avg_risk_score REAL,
                improvement_trend TEXT,
                email_sent BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Email logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                email_type TEXT,  -- 'weekly_summary', 'welcome', etc.
                subject TEXT,
                sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN,
                error_message TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        
        # Migrate existing database - add missing columns
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'password' not in columns:
            print("⚙️ Migrating database: Adding password column...")
            cursor.execute('ALTER TABLE users ADD COLUMN password TEXT')
            conn.commit()
        
        if 'last_login' not in columns:
            print("⚙️ Migrating database: Adding last_login column...")
            cursor.execute('ALTER TABLE users ADD COLUMN last_login TIMESTAMP')
            conn.commit()
            
        if 'total_logins' not in columns:
            print("⚙️ Migrating database: Adding total_logins column...")
            cursor.execute('ALTER TABLE users ADD COLUMN total_logins INTEGER DEFAULT 0')
            conn.commit()
        
        conn.close()
        print("✓ Database initialized successfully")
    
    def create_user_with_password(self, email, name=None, password=None, age_group=None):
        """Create a new user with password"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (email, password, name, age_group)
                VALUES (?, ?, ?, ?)
            ''', (email, password, name, age_group))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'user_id': user_id,
                'message': f'User {email} created successfully'
            }
        
        except sqlite3.IntegrityError:
            conn.close()
            return {
                'success': False,
                'error': 'Email already exists'
            }
        except Exception as e:
            conn.close()
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_login_stats(self, user_id):
        """Update user login statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users 
            SET last_login = CURRENT_TIMESTAMP, 
                total_logins = total_logins + 1 
            WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        return {'success': True}
        """Create a new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (email, name, age_group)
                VALUES (?, ?, ?)
            ''', (email, name, age_group))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'user_id': user_id,
                'message': f'User {email} created successfully'
            }
        
        except sqlite3.IntegrityError:
            conn.close()
            return {
                'success': False,
                'error': 'Email already exists'
            }
        except Exception as e:
            conn.close()
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_user_by_email(self, email):
        """Get user by email"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        return dict(user) if user else None
    
    def save_assessment(self, user_id, assessment_data, prediction_result, risk_scores):
        """Save assessment results"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO assessments 
                (user_id, assessment_data, prediction_result, risk_scores, is_healthy, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                json.dumps(assessment_data),
                json.dumps(prediction_result),
                json.dumps(risk_scores),
                prediction_result.get('is_healthy', False),
                prediction_result.get('confidence', 0.0)
            ))
            
            assessment_id = cursor.lastrowid
            
            # Update user's last assessment time
            cursor.execute('''
                UPDATE users SET last_assessment = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            return {
                'success': True,
                'assessment_id': assessment_id
            }
        
        except Exception as e:
            conn.close()
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_user_assessments(self, user_id, limit=10):
        """Get user's recent assessments"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM assessments 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        assessments = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts and parse JSON fields
        result = []
        for assessment in assessments:
            assessment_dict = dict(assessment)
            assessment_dict['assessment_data'] = json.loads(assessment_dict['assessment_data'])
            assessment_dict['prediction_result'] = json.loads(assessment_dict['prediction_result'])
            assessment_dict['risk_scores'] = json.loads(assessment_dict['risk_scores'])
            result.append(assessment_dict)
        
        return result
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total assessments
        cursor.execute('SELECT COUNT(*) as total FROM assessments WHERE user_id = ?', (user_id,))
        total_assessments = cursor.fetchone()['total']
        
        # Average risk score
        cursor.execute('''
            SELECT AVG(json_extract(risk_scores, '$.total_risk')) as avg_risk 
            FROM assessments WHERE user_id = ?
        ''', (user_id,))
        avg_risk = cursor.fetchone()['avg_risk'] or 0
        
        # Healthy vs unhealthy assessments
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN is_healthy = 1 THEN 1 ELSE 0 END) as healthy_count,
                SUM(CASE WHEN is_healthy = 0 THEN 1 ELSE 0 END) as unhealthy_count
            FROM assessments WHERE user_id = ?
        ''', (user_id,))
        health_stats = cursor.fetchone()
        
        # Recent trend (last 5 assessments)
        cursor.execute('''
            SELECT is_healthy, created_at FROM assessments 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 5
        ''', (user_id,))
        recent_assessments = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_assessments': total_assessments,
            'avg_risk_score': round(avg_risk, 2),
            'healthy_count': health_stats['healthy_count'] or 0,
            'unhealthy_count': health_stats['unhealthy_count'] or 0,
            'recent_trend': [dict(row) for row in recent_assessments]
        }
    
    def enable_weekly_emails(self, user_id, enabled=True):
        """Enable/disable weekly emails for user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET weekly_email_enabled = ? WHERE id = ?
        ''', (enabled, user_id))
        
        conn.commit()
        conn.close()
        
        return {'success': True}
    
    def update_user_profile(self, user_id, name=None, age_group=None, password=None):
        """Update user profile information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Build dynamic update query based on provided fields
            update_fields = []
            params = []
            
            if name is not None:
                update_fields.append("name = ?")
                params.append(name)
            
            if age_group is not None:
                update_fields.append("age_group = ?")
                params.append(age_group)
            
            if password is not None:
                update_fields.append("password = ?")
                params.append(password)
            
            if not update_fields:
                return {'success': False, 'error': 'No fields to update'}
            
            # Add user_id to params
            params.append(user_id)
            
            query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, params)
            
            if cursor.rowcount == 0:
                return {'success': False, 'error': 'User not found'}
            
            conn.commit()
            return {'success': True, 'message': 'Profile updated successfully'}
            
        except sqlite3.Error as e:
            conn.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            conn.close()
    
    def get_weekly_email_users(self):
        """Get users who have weekly emails enabled"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM users WHERE weekly_email_enabled = 1
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        return [dict(user) for user in users]
    
    def log_email(self, user_id, email_type, subject, success, error_message=None):
        """Log email sending attempt"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO email_logs (user_id, email_type, subject, success, error_message)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, email_type, subject, success, error_message))
        
        conn.commit()
        conn.close()
    
    def get_database_stats(self):
        """Get overall database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total users
        cursor.execute('SELECT COUNT(*) as count FROM users')
        total_users = cursor.fetchone()['count']
        
        # Total assessments
        cursor.execute('SELECT COUNT(*) as count FROM assessments')
        total_assessments = cursor.fetchone()['count']
        
        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) as count FROM assessments 
            WHERE created_at >= datetime('now', '-7 days')
        ''')
        recent_assessments = cursor.fetchone()['count']
        
        # Email stats
        cursor.execute('SELECT COUNT(*) as count FROM users WHERE weekly_email_enabled = 1')
        email_subscribers = cursor.fetchone()['count']
        
        conn.close()
        
        return {
            'total_users': total_users,
            'total_assessments': total_assessments,
            'recent_assessments': recent_assessments,
            'email_subscribers': email_subscribers,
            'database_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        }

# Initialize database instance
db = ScreenHealthDB()