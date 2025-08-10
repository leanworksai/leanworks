import logging
import datetime
import pytz
from typing import List, Dict, Optional
import requests
from msal import ConfidentialClientApplication, PublicClientApplication
import json

logger = logging.getLogger(__name__)

class OutlookTool:
    def __init__(self, client_id: str = None, client_secret: str = None, tenant_id: str = None, authority: str = None):
        """
        Initialize OutlookTool with Microsoft Graph API credentials.
        
        Args:
            client_id: Azure AD application client ID
            client_secret: Azure AD application client secret (for confidential client)
            tenant_id: Azure AD tenant ID
            authority: Azure AD authority URL
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.authority = authority or f"https://login.microsoftonline.com/{tenant_id}"
        self.scopes = ['https://graph.microsoft.com/.default']
        self.access_token = None
        
    def _authenticate(self):
        """Authenticate with Microsoft Graph API using client credentials flow."""
        try:
            if not all([self.client_id, self.client_secret, self.tenant_id]):
                logger.error("Missing required credentials: client_id, client_secret, and tenant_id")
                return False
            
            # Create confidential client application
            app = ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority
            )
            
            # Get token using client credentials flow
            result = app.acquire_token_for_client(scopes=self.scopes)
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                return True
            else:
                logger.error(f"Failed to acquire token: {result.get('error_description', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    @property
    def list_upcoming_meetings_property(self):
        description = """
        List all upcoming meetings for a user based on a specific date. The response will be a list of dictionaries, 
        each containing meeting details such as subject, start_time, end_time, attendees, location, and description. If it is empty, it means there is no upcoming meeting.
        This tool should be called when you need to retrieve a user's calendar information for scheduling or planning purposes.
        If no date is specified, it will default to today's date.
        The tool will return meetings from the specified date onwards.
        """
        return {
            "type": "custom",
            "name": "list_upcoming_meetings",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_email": {
                        "type": "string",
                        "description": "Email address of the user whose meetings to retrieve"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date to start looking for meetings from (YYYY-MM-DD format). Defaults to today if not specified."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of meetings to return. Defaults to 50 if not specified."
                    }
                },
                "required": ["user_email"]
            }
        }
    
    def list_upcoming_meetings(self, user_email: str, date: str = None, max_results: int = 50) -> List[Dict]:
        """
        List upcoming meetings for a user starting from a specific date.
        
        Args:
            user_email: Email address of the user
            date: Start date in YYYY-MM-DD format (defaults to today)
            max_results: Maximum number of meetings to return
            
        Returns:
            List of meeting dictionaries
        """
        logger.info(f"Executing list_upcoming_meetings for user: {user_email}, date: {date}, max_results: {max_results}")
        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}
            
            # Set default date to today if not provided
            if not date:
                date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # Convert date to datetime for comparison
            start_date = datetime.datetime.strptime(date, '%Y-%m-%d')
            start_datetime = start_date.isoformat() + 'Z'
            
            # Microsoft Graph API endpoint for user's calendar events
            graph_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/calendarView"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            # Set end date to 5 years from start date (1825 days) to comply with Graph API limits
            end_date = start_date + datetime.timedelta(days=1825)
            end_datetime = end_date.isoformat() + 'Z'
            
            params = {
                'startDateTime': start_datetime,
                'endDateTime': end_datetime,
                '$top': max_results,
                '$orderby': 'start/dateTime'
            }
            
            response = requests.get(graph_url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"Graph API error: {response.status_code} - {response.text}")
                return {"error": f"Graph API error: {response.status_code}"}
            
            events_data = response.json()
            events = events_data.get('value', [])
            meetings = []
            
            for event in events:
                # Check if it's a meeting (has attendees other than the organizer)
                attendees = event.get('attendees', [])
                organizer = event.get('organizer', {}).get('emailAddress', {}).get('address', '')
                
                # Filter out events that are not meetings
                meeting_attendees = [a.get('emailAddress', {}).get('address') for a in attendees if a.get('emailAddress', {}).get('address') != organizer]
                
                if meeting_attendees or event.get('isOnlineMeeting'):
                    # Parse start time to get day of week
                    start_time_str = event['start'].get('dateTime', event['start'].get('date'))
                    if 'T' in start_time_str:  # Has time component
                        start_dt = datetime.datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                        day_of_week = start_dt.strftime('%A')  # Full day name (Monday, Tuesday, etc.)
                    else:  # Date only
                        start_dt = datetime.datetime.strptime(start_time_str, '%Y-%m-%d')
                        day_of_week = start_dt.strftime('%A')
                    
                    meeting = {
                        'subject': event.get('subject', 'No Subject'),
                        'start_time': start_time_str,
                        'end_time': event['end'].get('dateTime', event['end'].get('date')),
                        'day_of_week': day_of_week,
                        'attendees': meeting_attendees,
                        'location': event.get('location', {}).get('displayName', 'No Location'),
                        'description': event.get('bodyPreview', 'No Description'),
                        'is_online': event.get('isOnlineMeeting', False),
                        'meeting_id': event.get('id')
                    }
                    meetings.append(meeting)
            
            return meetings
            
        except Exception as e:
            logger.error(f"Error retrieving meetings: {str(e)}")
            return {"error": f"Failed to retrieve meetings: {str(e)}"}
    
    @property
    def find_available_slots_property(self):
        description = """
        Find available meeting slots for a list of users within a specified time range, automatically considering each user's timezone.
        The response will be a list of available time slots where all specified users are free to meet. If it is empty, it means there is no available slot.
        This tool automatically detects each user's timezone from their calendar settings and searches for available slots 24/7 including weekends.
        The tool will check each user's calendar and timezone to identify overlapping free time periods.
        """
        return {
            "type": "custom",
            "name": "find_available_slots",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_emails": {
                        "type": "string",
                        "description": "Comma-separated list of email addresses of users to find available slots for"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date to search for available slots (YYYY-MM-DD format)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date to search for available slots (YYYY-MM-DD format)"
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration of the meeting in minutes. Defaults to 60 if not specified."
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone for the meeting (e.g., 'UTC', 'America/New_York', 'Europe/London'). Defaults to 'UTC'."
                    },
                    "work_hours": {
                        "type": "object",
                        "description": "Optional time range configuration (applied in each user's local timezone). If not provided, searches the full 24-hour period.",
                        "properties": {
                            "start_hour": {
                                "type": "integer",
                                "description": "Start hour of search period (0-23). Defaults to 0 (midnight)."
                            },
                            "end_hour": {
                                "type": "integer", 
                                "description": "End hour of search period (0-23). Defaults to 23 (11 PM)."
                            }
                        }
                    }
                },
                "required": ["user_emails", "start_date", "end_date"]
            }
        }
    
    def find_available_slots(self, user_emails: str, start_date: str, end_date: str, 
                           duration_minutes: int = 60, timezone: str = 'UTC', 
                           work_hours: Dict = None) -> List[Dict]:
        """
        Find available meeting slots for multiple users, automatically considering each user's timezone.
        Searches 24/7 including weekends for available slots.
        
        Args:
            user_emails: Comma-separated string of user email addresses
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            duration_minutes: Duration of the meeting in minutes
            timezone: Timezone for the meeting (e.g., 'UTC', 'America/New_York')
            work_hours: Optional dictionary with 'start_hour', 'end_hour' for custom time range (defaults to full 24-hour period)
            
        Returns:
            List of available time slots
        """
        logger.info(f"Executing find_available_slots for users: {user_emails}, start_date: {start_date}, end_date: {end_date}, duration_minutes: {duration_minutes}, timezone: {timezone}")
        try:
            if not self.access_token and not self._authenticate():
                return {"error": "Failed to authenticate with Microsoft Graph API"}
            
            # Parse comma-separated user emails
            user_emails_list = [email.strip() for email in user_emails.split(',')]
            
            # Set default work hours to full day if not provided
            if work_hours is None:
                work_hours = {
                    'start_hour': 0,
                    'end_hour': 23
                }
            
            # Get meeting timezone object
            meeting_tz = pytz.timezone(timezone)
            
            # Get each user's timezone from their calendar
            user_timezones = {}
            for user_email in user_emails_list:
                user_tz = self.get_user_timezone(user_email)
                user_timezones[user_email] = user_tz
                logger.info(f"User {user_email} timezone: {user_tz}")
            
            # Parse dates in meeting timezone
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
            # Make dates timezone-aware in meeting timezone
            start_dt = meeting_tz.localize(start_dt)
            end_dt = meeting_tz.localize(end_dt)
            
            # Get all users' busy times
            all_busy_times = []
            
            for user_email in user_emails_list:
                try:
                    # Get user's timezone
                    user_tz = user_timezones[user_email]
                    user_tz_obj = pytz.timezone(user_tz)
                    
                    # Convert meeting timezone dates to user's timezone for work hours calculation
                    start_dt_user_tz = start_dt.astimezone(user_tz_obj)
                    end_dt_user_tz = end_dt.astimezone(user_tz_obj)
                    
                    # Apply work hours in user's timezone
                    start_datetime_user_tz = start_dt_user_tz.replace(
                        hour=work_hours['start_hour'], 
                        minute=0, 
                        second=0, 
                        microsecond=0
                    )
                    end_datetime_user_tz = end_dt_user_tz.replace(
                        hour=work_hours['end_hour'], 
                        minute=59, 
                        second=59, 
                        microsecond=999999
                    )
                    
                    # Microsoft Graph API endpoint for user's calendar events
                    graph_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/calendarView"
                    
                    headers = {
                        'Authorization': f'Bearer {self.access_token}',
                        'Content-Type': 'application/json'
                    }
                    
                    # Convert to UTC for Graph API
                    start_utc = start_datetime_user_tz.astimezone(pytz.UTC)
                    end_utc = end_datetime_user_tz.astimezone(pytz.UTC)
                    
                    params = {
                        'startDateTime': start_utc.isoformat().replace('+00:00', 'Z'),
                        'endDateTime': end_utc.isoformat().replace('+00:00', 'Z'),
                        '$orderby': 'start/dateTime'
                    }
                    
                    response = requests.get(graph_url, headers=headers, params=params)
                    
                    if response.status_code != 200:
                        logger.warning(f"Could not retrieve calendar for {user_email}: {response.status_code}")
                        continue
                    
                    events_data = response.json()
                    events = events_data.get('value', [])
                    
                    # Convert events to busy time ranges
                    for event in events:
                        event_start = event['start'].get('dateTime')
                        event_end = event['end'].get('dateTime')
                        
                        if event_start and event_end:
                            # Parse datetime strings (Graph API returns UTC)
                            start_time = datetime.datetime.fromisoformat(event_start.replace('Z', '+00:00'))
                            end_time = datetime.datetime.fromisoformat(event_end.replace('Z', '+00:00'))
                            
                            # Ensure timezone awareness
                            if start_time.tzinfo is None:
                                start_time = start_time.replace(tzinfo=pytz.UTC)
                            if end_time.tzinfo is None:
                                end_time = end_time.replace(tzinfo=pytz.UTC)
                            
                            # Convert to meeting timezone for comparison
                            start_time_meeting_tz = start_time.astimezone(meeting_tz)
                            end_time_meeting_tz = end_time.astimezone(meeting_tz)
                            
                            all_busy_times.append({
                                'start': start_time_meeting_tz,
                                'end': end_time_meeting_tz,
                                'user': user_email,
                                'user_timezone': user_tz
                            })
                            
                except Exception as e:
                    logger.warning(f"Could not retrieve calendar for {user_email}: {str(e)}")
                    continue
            
            # Sort busy times by start time
            all_busy_times.sort(key=lambda x: x['start'])
            
            # Find available slots
            available_slots = []
            current_date = start_dt
            
            logger.info(f"Searching for available slots from {start_dt.date()} to {end_dt.date()}")
            logger.info(f"Total busy times collected: {len(all_busy_times)}")
            
            while current_date <= end_dt:
                # Create time slots for this day in meeting timezone (full 24-hour period)
                day_start = current_date.replace(
                    hour=work_hours['start_hour'], 
                    minute=0, 
                    second=0, 
                    microsecond=0
                )
                day_end = current_date.replace(
                    hour=work_hours['end_hour'], 
                    minute=59, 
                    second=59, 
                    microsecond=999999
                )
                
                # Find free time slots
                free_slots = self._find_free_slots_in_day(
                    day_start, day_end, all_busy_times, duration_minutes
                )
                
                for slot in free_slots:
                    # Convert slot times to meeting timezone for display
                    slot_start_meeting_tz = slot['start'].astimezone(meeting_tz)
                    slot_end_meeting_tz = slot['end'].astimezone(meeting_tz)
                    
                    # Get user timezone information for this slot
                    slot_user_timezones = {}
                    for user_email in user_emails_list:
                        user_tz = user_timezones[user_email]
                        slot_start_user_tz = slot['start'].astimezone(pytz.timezone(user_tz))
                        slot_end_user_tz = slot_start_user_tz + datetime.timedelta(minutes=duration_minutes)
                        slot_user_timezones[user_email] = {
                            'timezone': user_tz,
                            'start_time': slot_start_user_tz.strftime('%H:%M'),
                            'end_time': slot_end_user_tz.strftime('%H:%M')
                        }
                    
                    available_slots.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'day_of_week': current_date.strftime('%A'),  # Full day name (Monday, Tuesday, etc.)
                        'start_time': slot_start_meeting_tz.strftime('%H:%M'),
                        'end_time': slot_end_meeting_tz.strftime('%H:%M'),
                        'duration_minutes': duration_minutes,
                        'timezone': timezone,
                        'start_datetime_utc': slot['start'].astimezone(pytz.UTC).isoformat(),
                        'end_datetime_utc': slot['end'].astimezone(pytz.UTC).isoformat(),
                        'user_timezones': slot_user_timezones
                    })
                
                current_date += datetime.timedelta(days=1)
            
            logger.info(f"Found {len(available_slots)} total available slots")
            return available_slots
            
        except Exception as e:
            logger.error(f"Error finding available slots: {str(e)}")
            return {"error": f"Failed to find available slots: {str(e)}"}
    
    def _find_free_slots_in_day(self, day_start: datetime.datetime, day_end: datetime.datetime, 
                                busy_times: List[Dict], duration_minutes: int) -> List[Dict]:
        """Helper method to find free slots within a single day."""
        free_slots = []
        current_time = day_start
        
        # Filter busy times for this day
        day_busy_times = [
            bt for bt in busy_times 
            if bt['start'].date() == day_start.date() or bt['end'].date() == day_start.date()
        ]
        
        # Sort busy times for this day
        day_busy_times.sort(key=lambda x: x['start'])
        
        logger.info(f"Finding slots for {day_start.date()}: {len(day_busy_times)} busy times, work hours {day_start.strftime('%H:%M')}-{day_end.strftime('%H:%M')}")
        
        # Find all free time periods throughout the day
        while current_time < day_end:
            # Find the next busy time that starts after current time
            next_busy = None
            for busy_time in day_busy_times:
                if busy_time['start'] > current_time:
                    next_busy = busy_time
                    break
            
            # Calculate the end of this free period
            if next_busy is None:
                # No more busy times, free period extends to end of day
                free_period_end = day_end
            else:
                # Free period ends at the start of the next busy time
                free_period_end = next_busy['start']
            
            # Check if this free period is long enough for the meeting
            free_duration = (free_period_end - current_time).total_seconds() / 60  # in minutes
            if free_duration >= duration_minutes:
                # This free period is long enough, add it as a slot
                free_slots.append({
                    'start': current_time,
                    'end': free_period_end
                })
            
            # Move to after the busy period (or end of day if no more busy times)
            if next_busy is None:
                break
            else:
                current_time = next_busy['end']
        
        logger.info(f"Found {len(free_slots)} free slots for {day_start.date()}")
        return free_slots
    
    def _validate_timezone(self, timezone_str: str) -> bool:
        """Validate if a timezone string is valid."""
        try:
            pytz.timezone(timezone_str)
            return True
        except pytz.exceptions.UnknownTimeZoneError:
            return False
    
    def get_user_timezone(self, user_email: str) -> str:
        """
        Get the timezone for a specific user from their calendar settings.
        
        Args:
            user_email: Email address of the user
            
        Returns:
            Timezone string or 'UTC' if not found
        """
        try:
            if not self.access_token and not self._authenticate():
                return 'UTC'
            
            # Microsoft Graph API endpoint for user's calendar settings
            graph_url = f"https://graph.microsoft.com/v1.0/users/{user_email}/calendar"
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(graph_url, headers=headers)
            
            if response.status_code == 200:
                calendar_data = response.json()
                timezone = calendar_data.get('timeZone', 'UTC')
                return timezone if self._validate_timezone(timezone) else 'UTC'
            else:
                logger.warning(f"Could not retrieve calendar settings for {user_email}: {response.status_code}")
                return 'UTC'
                
        except Exception as e:
            logger.warning(f"Error getting timezone for {user_email}: {str(e)}")
            return 'UTC'
