#!/usr/bin/env python3
"""
Test script for the Outlook tool.
This demonstrates how to use the OutlookTool class to interact with Microsoft Graph API.
"""

import os
import datetime
from leanworks.agent.tools.outlook import OutlookTool
from dotenv import load_dotenv

load_dotenv()

def test_outlook_tool():
    """Test the Outlook tool functionality."""
    
    # Initialize the Outlook tool
    # You'll need to set these environment variables or provide them directly
    client_id = os.getenv('AD_CLIENT_ID')
    client_secret = os.getenv('AD_CLIENT_SECRET')
    tenant_id = os.getenv('AD_TENANT_ID')
    
    if not all([client_id, client_secret, tenant_id]):
        print("Please set the following environment variables:")
        print("AD_CLIENT_ID - Your Azure AD application client ID")
        print("AD_CLIENT_SECRET - Your Azure AD application client secret")
        print("AD_TENANT_ID - Your Azure AD tenant ID")
        return
    
    # Create Outlook tool instance
    outlook_tool = OutlookTool(
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id
    )
    
    # Test 1: List upcoming meetings for a user
    print("=== Testing list_upcoming_meetings ===")
    user_email = "yanfu@leanworks.ai"  # Replace with actual user email
    meetings = outlook_tool.list_upcoming_meetings(
        user_email=user_email,
        date=datetime.datetime.now().strftime('%Y-%m-%d'),  # Use today's date
        max_results=10
    )
    
    if isinstance(meetings, list):
        print(f"Found {len(meetings)} upcoming meetings:")
        for meeting in meetings:
            print(f"- {meeting['subject']} at {meeting['start_time']}")
    else:
        print(f"Error: {meetings}")
    
    # Test 2: Find available meeting slots with automatic timezone detection
    print("\n=== Testing find_available_slots with automatic timezone detection ===")
    user_emails = ["yanfu@leanworks.ai", "vijay@leanworks.ai", "qianwen@leanworks.ai"]  # Replace with actual emails
    
    # Use reasonable date range (next 7 days) to avoid Graph API limits
    today = datetime.datetime.now()
    start_date = today.strftime('%Y-%m-%d')
    end_date = (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Test with automatic timezone detection (work hours applied in each user's timezone)
    print("Testing automatic timezone detection:")
    print("Work hours will be applied in each user's local timezone from their calendar settings")
    
    # Test with different meeting durations
    for duration_minutes in [30, 60, 120]:
        print(f"\n--- Testing {duration_minutes}-minute meetings ---")
        
        available_slots = outlook_tool.find_available_slots(
            user_emails=",".join(user_emails),
            start_date=start_date,
            end_date=end_date,
            duration_minutes=duration_minutes,
            timezone='America/Los_Angeles',  # Meeting timezone
            work_hours={
                'start_hour': 9,
                'end_hour': 17
                # No timezone specified - will use each user's detected timezone
            }
        )
        
        if isinstance(available_slots, list):
            print(f"Found {len(available_slots)} available slots for {duration_minutes}-minute meetings:")
            
            # Group slots by date to see how many per day
            slots_by_date = {}
            for slot in available_slots:
                date = slot['date']
                if date not in slots_by_date:
                    slots_by_date[date] = []
                slots_by_date[date].append(slot)
            
            # Show first 2 days
            for date in sorted(slots_by_date.keys())[:2]:
                day_slots = slots_by_date[date]
                print(f"\n{date} ({len(day_slots)} slots):")
                for slot in day_slots:
                    # Calculate duration of this slot
                    start_time = datetime.datetime.strptime(slot['start_time'], '%H:%M')
                    end_time = datetime.datetime.strptime(slot['end_time'], '%H:%M')
                    duration_hours = (end_time - start_time).total_seconds() / 3600
                    
                    print(f"  - {slot['start_time']} to {slot['end_time']} ({slot['timezone']}) - {duration_hours:.1f} hours")
        else:
            print(f"Error: {available_slots}")
    
    # Test 3: Get user timezone
    print("\n=== Testing get_user_timezone ===")
    for user_email in user_emails:
        timezone = outlook_tool.get_user_timezone(user_email)
        print(f"Timezone for {user_email}: {timezone}")

if __name__ == "__main__":
    test_outlook_tool()
