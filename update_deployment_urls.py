#!/usr/bin/env python3
"""
ScreenHealth AI - URL Update Script for Deployment
This script updates all localhost URLs to your deployed app URL
"""

import os
import re

def update_urls_for_deployment():
    print("🚀 ScreenHealth AI - Deployment URL Updater")
    print("=" * 50)
    
    # Get the deployed URL from user
    deployed_url = input("Enter your deployed app URL (e.g., https://your-app.herokuapp.com): ").strip()
    
    if not deployed_url:
        print("❌ No URL provided. Exiting.")
        return
    
    # Remove trailing slash if present
    deployed_url = deployed_url.rstrip('/')
    
    print(f"🔄 Updating URLs to: {deployed_url}")
    print()
    
    # Files to update
    frontend_files = [
        'frontend/assessment.html',
        'frontend/dashboard.html', 
        'frontend/login.html',
        'frontend/results.html'
    ]
    
    updated_count = 0
    
    for file_path in frontend_files:
        if os.path.exists(file_path):
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Count occurrences before replacement
                localhost_count = len(re.findall(r'http://localhost:5000', content))
                
                if localhost_count > 0:
                    # Replace localhost URLs with deployed URL
                    content = re.sub(r'http://localhost:5000', deployed_url, content)
                    
                    # Write back to file
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(content)
                    
                    print(f"✅ Updated {file_path} ({localhost_count} URLs changed)")
                    updated_count += 1
                else:
                    print(f"ℹ️  {file_path} - No localhost URLs found")
            
            except Exception as e:
                print(f"❌ Error updating {file_path}: {e}")
        else:
            print(f"⚠️  {file_path} - File not found")
    
    print()
    print("=" * 50)
    if updated_count > 0:
        print(f"🎉 Successfully updated {updated_count} files!")
        print()
        print("📋 Next steps:")
        print("1. Test your app locally to make sure it still works")
        print("2. Commit the changes: git add . && git commit -m 'Update URLs for deployment'")
        print("3. Deploy: git push heroku main (or your deployment method)")
        print("4. Test your deployed app!")
    else:
        print("ℹ️  No files needed updating.")

def revert_to_localhost():
    """Revert URLs back to localhost for local development"""
    print("🔄 Reverting URLs back to localhost...")
    
    frontend_files = [
        'frontend/assessment.html',
        'frontend/dashboard.html', 
        'frontend/login.html',
        'frontend/results.html'
    ]
    
    reverted_count = 0
    
    for file_path in frontend_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Find any https URLs and replace with localhost
                https_urls = re.findall(r'https://[^/]+\.(?:herokuapp|railway|render|pythonanywhere)\.(?:com|app)', content)
                
                if https_urls:
                    for url in set(https_urls):  # Remove duplicates
                        content = content.replace(url, 'http://localhost:5000')
                    
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(content)
                    
                    print(f"✅ Reverted {file_path}")
                    reverted_count += 1
            
            except Exception as e:
                print(f"❌ Error reverting {file_path}: {e}")
    
    if reverted_count > 0:
        print(f"🎉 Reverted {reverted_count} files back to localhost!")
    else:
        print("ℹ️  No files needed reverting.")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Update URLs for deployment")
    print("2. Revert URLs to localhost")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        update_urls_for_deployment()
    elif choice == "2":
        revert_to_localhost()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")