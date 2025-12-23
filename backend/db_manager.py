"""
Database Management Script for ScreenHealth AI
Run this script to manage your SQLite database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import db
import json

def show_stats():
    """Show database statistics"""
    stats = db.get_database_stats()
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)
    print(f"Total Users: {stats['total_users']}")
    print(f"Total Assessments: {stats['total_assessments']}")
    print(f"Recent Assessments (7 days): {stats['recent_assessments']}")
    print(f"Email Subscribers: {stats['email_subscribers']}")
    print(f"Database Size: {stats['database_size']} bytes")
    print("="*50)

def list_users():
    """List all users"""
    import sqlite3
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, email, name, age_group, created_at, last_assessment, weekly_email_enabled
        FROM users ORDER BY created_at DESC
    ''')
    
    users = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*80)
    print("ALL USERS")
    print("="*80)
    print(f"{'ID':<5} {'Email':<25} {'Name':<15} {'Age Group':<12} {'Created':<12} {'Weekly Email'}")
    print("-"*80)
    
    for user in users:
        created = user['created_at'][:10] if user['created_at'] else 'N/A'
        weekly = 'Yes' if user['weekly_email_enabled'] else 'No'
        print(f"{user['id']:<5} {user['email']:<25} {user['name'] or 'N/A':<15} {user['age_group'] or 'N/A':<12} {created:<12} {weekly}")
    
    print("="*80)

def show_user_details(email):
    """Show detailed user information"""
    user = db.get_user_by_email(email)
    if not user:
        print(f"User {email} not found!")
        return
    
    stats = db.get_user_stats(user['id'])
    assessments = db.get_user_assessments(user['id'], 5)
    
    print("\n" + "="*60)
    print(f"USER DETAILS: {email}")
    print("="*60)
    print(f"ID: {user['id']}")
    print(f"Name: {user['name'] or 'N/A'}")
    print(f"Age Group: {user['age_group'] or 'N/A'}")
    print(f"Created: {user['created_at']}")
    print(f"Last Assessment: {user['last_assessment'] or 'Never'}")
    print(f"Weekly Emails: {'Enabled' if user['weekly_email_enabled'] else 'Disabled'}")
    
    print(f"\nSTATISTICS:")
    print(f"Total Assessments: {stats['total_assessments']}")
    print(f"Average Risk Score: {stats['avg_risk_score']}")
    print(f"Healthy Assessments: {stats['healthy_count']}")
    print(f"Unhealthy Assessments: {stats['unhealthy_count']}")
    
    if assessments:
        print(f"\nRECENT ASSESSMENTS (Last 5):")
        for i, assessment in enumerate(assessments, 1):
            status = "Healthy" if assessment['is_healthy'] else "Needs Improvement"
            risk = json.loads(assessment['risk_scores'])['total_risk']
            date = assessment['created_at'][:16]
            print(f"  {i}. {date} - {status} (Risk: {risk:.1f})")
    
    print("="*60)

def reset_database():
    """Reset database (WARNING: Deletes all data!)"""
    confirm = input("⚠️  WARNING: This will delete ALL data! Type 'CONFIRM' to proceed: ")
    if confirm != 'CONFIRM':
        print("Operation cancelled.")
        return
    
    import os
    if os.path.exists(db.db_path):
        os.remove(db.db_path)
        print("Database deleted.")
    
    db.init_database()
    print("Database reset successfully!")

def show_email_logs():
    """Show recent email logs"""
    import sqlite3
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            e.id, e.email_type, e.subject, e.sent_at, e.success, e.error_message,
            u.email, u.name
        FROM email_logs e
        JOIN users u ON e.user_id = u.id
        ORDER BY e.sent_at DESC
        LIMIT 20
    ''')
    
    logs = cursor.fetchall()
    conn.close()
    
    print("\n" + "="*100)
    print("EMAIL LOGS (Last 20)")
    print("="*100)
    print(f"{'ID':<5} {'Email':<25} {'Type':<15} {'Subject':<30} {'Sent':<12} {'Status'}")
    print("-"*100)
    
    for log in logs:
        sent_date = log['sent_at'][:10] if log['sent_at'] else 'N/A'
        status = 'Success' if log['success'] else 'Failed'
        subject = (log['subject'][:27] + '...') if len(log['subject']) > 30 else log['subject']
        print(f"{log['id']:<5} {log['email']:<25} {log['email_type']:<15} {subject:<30} {sent_date:<12} {status}")
    
    print("="*100)
    """Export all data to JSON"""
    import sqlite3
    conn = db.get_connection()
    
    # Export users
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = [dict(row) for row in cursor.fetchall()]
    
    # Export assessments
    cursor.execute('SELECT * FROM assessments')
    assessments = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    export_data = {
        'export_date': str(datetime.now()),
        'users': users,
        'assessments': assessments
    }
    
    filename = f"screenhealth_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Data exported to: {filename}")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*50)
        print("SCREENHEALTH AI - DATABASE MANAGER")
        print("="*50)
        print("1. Show Statistics")
        print("2. List All Users")
        print("3. Show User Details")
        print("4. Show Email Logs")
        print("5. Send Test Email")
        print("6. View Database Tables")
        print("7. Export Data")
        print("8. Reset Database (⚠️  DANGER)")
        print("0. Exit")
        print("="*50)
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            show_stats()
        elif choice == '2':
            list_users()
        elif choice == '3':
            email = input("Enter user email: ").strip()
            if email:
                show_user_details(email)
        elif choice == '4':
            show_email_logs()
        elif choice == '5':
            send_test_email()
        elif choice == '6':
            # Quick database view
            import sqlite3
            conn = db.get_connection()
            cursor = conn.cursor()
            
            print("\n📊 DATABASE TABLES")
            print("=" * 40)
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"📋 {table_name}: {count} records")
            
            conn.close()
        elif choice == '7':
            export_data()
        elif choice == '8':
            reset_database()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    from datetime import datetime
    main()

def export_data():
    """Export all data to JSON"""
    import sqlite3
    conn = db.get_connection()
    
    # Export users
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = [dict(row) for row in cursor.fetchall()]
    
    # Export assessments
    cursor.execute('SELECT * FROM assessments')
    assessments = [dict(row) for row in cursor.fetchall()]
    
    # Export email logs
    cursor.execute('SELECT * FROM email_logs')
    email_logs = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    export_data = {
        'export_date': str(datetime.now()),
        'users': users,
        'assessments': assessments,
        'email_logs': email_logs
    }
    
    filename = f"screenhealth_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Data exported to: {filename}")

def send_test_email():
    """Send a test email"""
    email = input("Enter email address to test: ").strip()
    if not email:
        print("Email required!")
        return
    
    # Import the send_email function
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from app import send_email
        
        test_html = """
        <h1>🧠 Test Email from ScreenHealth AI</h1>
        <p>This is a test email to verify your SMTP configuration.</p>
        <p>If you're receiving this, your email system is working correctly!</p>
        <p><strong>Timestamp:</strong> {}</p>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        print("Sending test email...")
        success = send_email(email, "ScreenHealth AI - Test Email", test_html)
        
        if success:
            print(f"✅ Test email sent successfully to {email}")
        else:
            print(f"❌ Failed to send test email to {email}")
            print("Check your .env file configuration:")
            print("- SMTP_USERNAME=your-email@gmail.com")
            print("- SMTP_PASSWORD=your-app-password")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Flask app is properly configured.")