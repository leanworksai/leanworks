from google.cloud import bigquery
import logging
import datetime
import uuid

logger = logging.getLogger(__name__)

class ProjectTool:
    bq_client = bigquery.Client.from_service_account_json("gcp_credential.json")
    
    @property
    def list_projects_property(self):
        description = """
        List all projects for a user. The response will be a list of dictionaries, each containing the project details such as project_id, project_name, description, collaborators, created_by and created_ts.
        If the user id is not given or is not in the format of email address, you need to call this tool without setting the user_id. DON'T invent a user id or email address.
        Sometimes, a user will come in asking for projects for a specific user. In this case, you MUST call this tool with the user_id set to the email address of the other user, instead of your own email address.
        Sometimes, a user will come in asking for projects for multiple users. In this case, you need to call this tool without setting the user_id.
        """
        return {
            "type": "custom",
            "name": "list_projects",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string", 
                        "description": "User email address"
                    }
                }
            }
        }
    def list_projects(self, user_id: str = None):
        try:
            if user_id:
                query = f'''
                SELECT * EXCEPT(last_n_days)
                FROM `leanworks.leanworks.project_config`
                WHERE collaborators LIKE '%{user_id}%'
                '''
            else:
                query = f'''
                SELECT * EXCEPT(last_n_days)
                FROM `leanworks.leanworks.project_config`
                '''
            logger.info(f"Executing BQ query in list_projects: {query}")
            query_job = self.bq_client.query(query)
            results = query_job.result()
            projects = []
            for row in results:
                project = dict(row)
                # Convert created_ts (int) to date string
                if 'created_ts' in project and project['created_ts']:
                    timestamp = int(project['created_ts'])
                    date_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    project['created_ts'] = date_str
                projects.append(project)
            return projects
        except Exception as e:
            logger.warning(f"Error retrieving projects for user {user_id}: {str(e)}")
            return []
        
    @property
    def list_tasks_property(self):
        description = """
        List all tasks for a user or project. The response will be a list of dictionaries, each containing the task details such as project_id, user_id, task_id, created_at, updated_at, task_name, status, description, priority, deadline and reason.
        If the user id is not given, you need to call this tool without setting the user_id. DON'T invent a user id or email address.
        If the user id is given in terms of first name or last name, call list_users tool to fetch the user_id and then call this tool with the user_id.
        Sometimes, a user will come in asking for tasks for a specific user. In this case, you MUST call this tool with the user_id set to the email address of the other user, instead of your own email address.
        Sometimes, a user will come in asking for tasks for multiple users. In this case, you need to call this tool without setting the user_id.
        If the project is not given, you need to list tasks across all projects for the user.
        """
        return {
            "type": "custom",
            "name": "list_tasks",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string", 
                        "description": "user_id is the email address of the user you are asked to list tasks for. It is not display name or uuid."
                    },
                    "project_id": {
                        "type": "string", 
                        "description": "Project identifier"
                    }
                }
            }
        }
    def list_tasks(self, user_id: str = None, project_id: str = None):
        try:
            # Build WHERE clause dynamically based on provided parameters
            where_conditions = []
            if user_id:
                where_conditions.append(f"user_id = '{user_id}'")
            if project_id:
                where_conditions.append(f"project_id = '{project_id}'")
            
            # Combine conditions with AND if both are provided
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f'''
            SELECT *
            FROM `leanworks.leanworks.tasks` 
            WHERE {where_clause}
            '''
            logger.info(f"Executing BQ query in list_tasks: {query}")
            query_results = self.bq_client.query(query)
            tasks = []
            for row in query_results:
                task = dict(row)
                # Convert timestamps (float) to date strings
                if 'created_at' in task and task['created_at']:
                    task['created_at'] = datetime.datetime.fromtimestamp(float(task['created_at'])).strftime('%Y-%m-%d %H:%M:%S')
                if 'updated_at' in task and task['updated_at']:
                    task['updated_at'] = datetime.datetime.fromtimestamp(float(task['updated_at'])).strftime('%Y-%m-%d %H:%M:%S')
                if 'deadline' in task and task['deadline']:
                    task['deadline'] = datetime.datetime.fromtimestamp(float(task['deadline'])).strftime('%Y-%m-%d %H:%M:%S')
                tasks.append(task)
            return tasks
        except Exception as e:
            # Handle table not found or any other exceptions
            logger.warning(f"Error retrieving tasks for user {user_id} and project {project_id}: {str(e)}")
            # Return empty list instead of propagating the exception
            return []

    @property
    def list_progress_updates_property(self):
        description = """
        List all progress updates for a user or project within date range. Leave date field empty if not specified in the query. 
        If the query contains words like "recent" or "latest" without any specific time frame, interpret this as looking back 1 week from today.
        The response will be a list of dictionaries, each containing the progress update details such as project_id, user_id, task_id, created_at, updated_at, task_name, status, description, priority, deadline and reason.
        If the user id is not given, you need to call this tool without setting the user_id. DON'T invent a user id or email address.
        If the user id is given in terms of first name or last name, call list_users tool to fetch the user_id and then call this tool with the user_id.
        Sometimes, a user will come in asking for progress updates for a specific user. In this case, you MUST call this tool with the user_id set to the email address of the other user, instead of your own email address.
        Sometimes, a user will come in asking for progress updates for multiple users. In this case, you need to call this tool without setting the user_id.
        If the project is not given, you need to list progress updates across all projects for the user.
        If start_date and end_date are not given, you need to list progress updates for whole time period.
        """
        return {
            "type": "custom",
            "name": "list_progress_updates",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string", 
                        "description": "User email address"
                    },
                    "project_id": {
                        "type": "string", 
                        "description": "Project identifier"
                    },
                    "start_date": {
                        "type": "string", 
                        "description": "Start date in YYYY-MM-DD format."
                    },
                    "end_date": {
                        "type": "string", 
                        "description": "End date in YYYY-MM-DD format."
                    }
                }
            }
        }
    def list_progress_updates(self, user_id: str = None, project_id: str = None, start_date: str = None, end_date: str = None):
        try:
            # Set default start_date to one week ago if not provided, using UTC
            if not start_date:
                one_week_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
                start_date = one_week_ago.strftime('%Y-%m-%d')
            
            # Parse date strings only if they don't follow the YYYY-MM-DD format
            formatted_start_date = start_date
            formatted_end_date = end_date
            
            # Function to check and format date if needed
            def ensure_date_format(date_str):
                if not date_str:
                    return None
                    
                # Check if already in YYYY-MM-DD format
                try:
                    datetime.datetime.strptime(date_str, '%Y-%m-%d')
                    return date_str  # Already in correct format
                except ValueError:
                    # Try to parse and convert to YYYY-MM-DD format
                    try:
                        # This will attempt to parse various date formats
                        parsed_date = datetime.datetime.strptime(date_str, '%m/%d/%Y')
                        return parsed_date.strftime('%Y-%m-%d')
                    except ValueError:
                        try:
                            parsed_date = datetime.datetime.strptime(date_str, '%d-%m-%Y')
                            return parsed_date.strftime('%Y-%m-%d')
                        except ValueError:
                            logger.warning(f"Could not parse date: {date_str}. Using as is.")
                            return date_str
            
            # Format dates if needed
            formatted_start_date = ensure_date_format(start_date)
            formatted_end_date = ensure_date_format(end_date)
            
            # Build WHERE clause dynamically based on provided parameters
            where_conditions = []
            
            if user_id:
                where_conditions.append(f"user_id = '{user_id}'")
            if project_id:
                where_conditions.append(f"project_id = '{project_id}'")
            if formatted_start_date:
                where_conditions.append(f"date_id >= '{formatted_start_date}'")
            if formatted_end_date:
                where_conditions.append(f"date_id <= '{formatted_end_date}'")
            
            # Combine conditions with AND if any are provided, otherwise use 1=1 to select all
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f'''
            SELECT *
            FROM `leanworks.leanworks.updates`
            WHERE {where_clause}
            '''
            logger.info(f"Executing BQ query in list_progress_updates: {query}")
            query_results = self.bq_client.query(query)
            
            # Convert date objects and timestamps to strings to make them JSON serializable
            results = []
            for row in query_results:
                row_dict = dict(row)
                for key, value in row_dict.items():
                    if isinstance(value, (datetime.date, datetime.datetime)):
                        row_dict[key] = value.isoformat()
                    elif key == "ts" and isinstance(value, (int, float)):
                        # Convert timestamp (float) to date string
                        row_dict[key] = datetime.datetime.fromtimestamp(float(value)).strftime('%Y-%m-%d %H:%M:%S')
                results.append(row_dict)
                
            return results
        except Exception as e:
            logger.warning(f"Error retrieving progress updates for user {user_id}: {str(e)}")
            return []
    @property
    def add_task_property(self):
        description = """
        Add a new task to a project. Any field, if not explicitly provided, will need to be inferred from the context.
        The response will be a dictionary with the task details such as project_id, user_id, task_id, created_at, updated_at, task_name, status, description, priority, deadline and reason.
        """
        return {
            "type": "custom",
            "name": "add_task",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Project identifier"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "Summary of the task. Limit each one to maximum 100 characters."
                    },
                    "description": {
                        "type": "string",
                        "description": "Full description of the task."
                    },
                    "deadline": {
                        "type": "string",
                        "description": "Deadline in YYYY-MM-DD format. Leave empty if it cannot be determined."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User email address to assign task to"
                    },
                    "priority": {
                        "type": "string",
                        "description": '"high", "medium", and "low". The default priority is "medium". It will be given "high" priority if there is a clear indication that the task is critical or time-sensitive. If there is a clear indication that the task is not critical or time-sensitive, it will be given "low" priority.'
                    },
                    "reason": {
                        "type": "string",
                        "description": "Explain the reason why the task is chosen."
                    }
                },
                "required": ["project_id", "task_name", "description", "deadline", "user_id", "priority", "reason"]
            }
        }
    def add_task(self, project_id: str, task_name: str, description: str = None, 
                 deadline: str = None, user_id: str = None, 
                 priority: str = "medium", reason: str = None):
        """
        Add a new task to the tasks table.
        
        Args:
            project_id: The ID of the project this task belongs to
            task_name: The name of the task
            description: Optional description of the task
            deadline: Optional deadline in YYYY-MM-DD format (will be converted to timestamp)
            user_id: Optional user ID of the person assigned to the task
            status: Task status (default: "Open")
            priority: Task priority (default: "medium")
            reason: Optional reason for the task
            
        Returns:
            Dictionary with task details if successful, error message otherwise
        """
        try:
            # Format deadline if provided
            formatted_deadline = None
            if deadline:
                try:
                    # Convert deadline to timestamp
                    parsed_date = datetime.datetime.strptime(deadline, '%Y-%m-%d')
                    formatted_deadline = parsed_date.timestamp()
                except ValueError:
                    logger.warning(f"Invalid deadline format: {deadline}. Expected YYYY-MM-DD.")
                    return {"error": "Invalid deadline format. Expected YYYY-MM-DD."}
            
            # Generate a unique task ID
            task_id = str(uuid.uuid4())
            
            # Create timestamps for created_at and updated_at
            current_timestamp = datetime.datetime.now().timestamp()
            
            # Prepare the query
            query = f"""
            INSERT INTO `leanworks.leanworks.tasks`
            (project_id, user_id, task_id, created_at, updated_at, task_name, status, description, priority, deadline, reason)
            VALUES
            ('{project_id}', 
            {f"'{user_id}'" if user_id else "NULL"}, 
            '{task_id}', 
            {current_timestamp}, 
            {current_timestamp}, 
            '{task_name}', 
            'open', 
            {f"'{description}'" if description else "NULL"}, 
            '{priority}', 
            {formatted_deadline if formatted_deadline else "NULL"}, 
            {f"'{reason}'" if reason else "NULL"})
            """
            
            # Execute the query
            logger.info(f"Executing BQ query in add_task: {query}")
            query_job = self.bq_client.query(query)
            query_job.result()  # Wait for the query to complete
            
            # Return the task details
            return {
                "task_id": task_id,
                "project_id": project_id,
                "user_id": user_id,
                "task_name": task_name,
                "description": description,
                "deadline": formatted_deadline,
                "status": "open",
                "priority": priority,
                "reason": reason,
                "created_at": current_timestamp,
                "updated_at": current_timestamp
            }
            
        except Exception as e:
            logger.warning(f"Error adding task to project {project_id}: {str(e)}")
            return {"error": f"Failed to add task: {str(e)}"}
        
    @property
    def list_users_property(self):
        description = """
        List all users in the team. The response will be a list of dictionaries, each containing the user details such as user_id, first_name and last_name.
        This tool can be used to search for the user_id if the user_id is not in the email address format.
        """
        return {
            "type": "custom",
            "name": "list_users",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    }
                }
            }
    def list_users(self):
        query = "SELECT * EXCEPT(user_id), alias_email as user_id FROM `leanworks.leanworks.user_config`"
        logger.info(f"Executing BQ query in list_users: {query}")
        query_job = self.bq_client.query(query)
        results = query_job.result()
        return [dict(row) for row in results]