from google.cloud import bigquery
import logging
import datetime
import uuid

logger = logging.getLogger(__name__)

class ProjectTool:
    def __init__(self, bq_client):
        """
        Initialize ProjectTool with a BigQuery client that contains dataset_id.
        
        Args:
            bq_client: BigQuery client object that has dataset_id attribute
        """
        self.bq_client = bq_client
    
    @property
    def list_projects_property(self):
        description = """
        List all projects for a user. The response will be a list of dictionaries, each containing the project details such as project_id, project_name, description, collaborators, created_by and created_ts.
        This tool should be called to retrieve project information when project details are needed to answer the question but is lacking in context, 
        If the user id is not given or is not in the format of email address, you need to call this tool without setting the user_id. DON'T invent a user id or email address.
        Sometimes, a user will come in asking for projects for a specific user. In this case, you MUST call this tool with the user_id set to the email address of the other user, instead of your own email address.
        Sometimes, a user will come in asking for projects for multiple users. In this case, you need to call this tool without setting the user_id.
        Since this tool only provide basic project information, you are recommended to call search_knowledge tool after if you want to dive deeper into a specific project.
        project_id can be used to link the projects to tasks and progress updates.
        You might need to call list_tasks or list_progress_updates before or after to understand the relationship among projects, tasks and progress updates through project_id.       
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
                        "description": "user_id is the email address of the user you are asked to list tasks for. It is not display name or uuid."
                    }
                }
            }
        }
    def list_projects(self, user_id: str = None):
        try:
            if user_id:
                query = f'''
                SELECT * EXCEPT(last_n_days)
                FROM `leanworks.{self.bq_client.dataset_id}.project_config`
                WHERE collaborators LIKE '%{user_id}%'
                '''
            else:
                query = f'''
                SELECT * EXCEPT(last_n_days)
                FROM `leanworks.{self.bq_client.dataset_id}.project_config`
                '''
            logger.info(f"Executing BQ query in list_projects: {query}")
            query_job = self.bq_client.client.query(query)
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
        This tool should be called to retrieve task information when task details are needed to answer the question but is lacking in context.
        This tool can also be used to retrieve a specific task by setting the task_id, if the task_id is explicitly provided.
        Sometimes, a user will come in asking for tasks for a specific user. In this case, you MUST call this tool with the user_id set to the email address of the other user, instead of your own email address.
        Sometimes, a user will come in asking for tasks for multiple users. In this case, you need to call this tool without setting the user_id.
        If the project id is not given, you need to list tasks across all projects for the user by calling this tool without setting the project_id.
        If the user id is not given, you need to call this tool without setting the user_id. DON'T invent a user id or email address.
        If the user id is given in terms of first name or last name, call list_users tool to fetch the user_id and then call this tool with the user_id.
        Since this tool only provide basic task information, you are recommended to call search_knowledge tool after if you want to dive deeper into a specific task.
        project_id can be used to link the tasks to projects.
        task_id can be used to link the tasks to progress updates.
        You might need to call list_projects or list_progress_updates before or after to understand the relationship among projects, tasks and progress updates through project_id or task_id.
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
                        "description": "Project identifier. This is not the project name or task name."
                    },
                    "task_id": {
                        "type": "string", 
                        "description": "Task identifier. This is not the task name or project name."
                    }
                }
            }
        }
    def list_tasks(self, user_id: str = None, project_id: str = None, task_id: str = None):
        try:
            # Build WHERE clause dynamically based on provided parameters
            where_conditions = []
            if user_id:
                where_conditions.append(f"user_id = '{user_id}'")
            if project_id:
                where_conditions.append(f"project_id = '{project_id}'")
            if task_id:
                where_conditions.append(f"task_id = '{task_id}'")
            # Combine conditions with AND if both are provided
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f'''
            SELECT *
            FROM `leanworks.{self.bq_client.dataset_id}.tasks` 
            WHERE {where_clause}
            '''
            logger.info(f"Executing BQ query in list_tasks: {query}")
            query_results = self.bq_client.client.query(query)
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
        List all progress updates for a user or project within date range. The response will be a list of dictionaries, each containing the progress update details such as project_id, user_id, task_id, created_at, updated_at, task_name, status, description, priority, deadline and reason.
        This tool should be called to retrieve progress update information when progress update details are needed to answer the question but is lacking in context.
        This tool can also be used to retrieve a specific progress update by setting the update_id, if the update_id is explicitly provided.
        If the query contains words like "recent" or "latest" without any specific time frame, interpret this as looking back 7 days from today.
        If the user id is not given, you need to call this tool without setting the user_id. DON'T invent a user id or email address.
        If the user id is given in terms of first name or last name, call list_users tool to fetch the user_id and then call this tool with the user_id.
        Sometimes, a user will come in asking for progress updates for a specific user. In this case, you MUST call this tool with the user_id set to the email address of the other user, instead of your own email address.
        Sometimes, a user will come in asking for progress updates for multiple users. In this case, you need to call this tool without setting the user_id.
        If the project id is not given, you need to list progress updates across all projects for the user by calling this tool without setting the project_id.
        If start_date or end_date is not explicitly specified or inferred from the query, you need to list progress updates for last 7 days. Don't it them up.
        Since this tool only provide basic progress update information, you are recommended to call search_knowledge tool after if you want to dive deeper into a specific progress update.
        project_id can be used to link the progress updates to projects.
        associated_tasks (a list of task_id) can be used to link the progress updates to tasks.
        You might need to call list_projects or list_tasks before or after to understand the relationship among projects, tasks and progress updates through project_id or task_id.
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
                        "description": "user_id is the email address of the user you are asked to list tasks for. It is not display name or uuid."
                    },
                    "project_id": {
                        "type": "string", 
                        "description": "Project identifier. This is not the project name or task name."
                    },
                    "update_id": {
                        "type": "string", 
                        "description": "Progress update identifier. This is not the progress update details, task name or project name."
                    },
                    "start_date": {
                        "type": "string", 
                        "description": "Start date in YYYY-MM-DD format. This is for finding progress updates that are made after the start date."
                    },
                    "end_date": {
                        "type": "string", 
                        "description": "End date in YYYY-MM-DD format. This is for finding progress updates that are made before the end date."
                    }
                }
            }
        }
    def list_progress_updates(self, user_id: str = None, project_id: str = None, update_id: str = None, start_date: str = None, end_date: str = None):
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
            if update_id:
                where_conditions.append(f"update_id = '{update_id}'")
            if formatted_start_date:
                where_conditions.append(f"date_id >= '{formatted_start_date}'")
            if formatted_end_date:
                where_conditions.append(f"date_id <= '{formatted_end_date}'")
            
            # Combine conditions with AND if any are provided, otherwise use 1=1 to select all
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f'''
            SELECT *
            FROM `leanworks.{self.bq_client.dataset_id}.updates`
            WHERE {where_clause}
            '''
            logger.info(f"Executing BQ query in list_progress_updates: {query}")
            query_results = self.bq_client.client.query(query)
            
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
        You might need to call list_projects before this tool to fetch project id that the task belongs to, if it is not provided in the context.
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
                        "description": "Project identifier. This is not the project name or task name."
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
            INSERT INTO `leanworks.{self.bq_client.dataset_id}.tasks`
            (project_id, user_id, task_id, created_at, updated_at, task_name, status, description, priority, deadline, reason)
            VALUES
            ('{project_id}', 
            {f"'{user_id}'" if user_id else "NULL"}, 
            '{task_id}', 
            {current_timestamp}, 
            {current_timestamp}, 
            '{task_name}', 
            'to_do', 
            {f"'{description}'" if description else "NULL"}, 
            '{priority}', 
            {formatted_deadline if formatted_deadline else "NULL"}, 
            {f"'{reason}'" if reason else "NULL"})
            """
            
            # Execute the query
            logger.info(f"Executing BQ query in add_task: {query}")
            query_job = self.bq_client.client.query(query)
            query_job.result()  # Wait for the query to complete
            
            # Return the task details
            return {
                "task_id": task_id,
                "project_id": project_id,
                "user_id": user_id,
                "task_name": task_name,
                "description": description,
                "deadline": formatted_deadline,
                "status": "to_do",
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
        This tool can be used to verify the user's id or try to find the user's email address.
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
        query = f"SELECT * EXCEPT(user_id), alias_email as user_id FROM `leanworks.{self.bq_client.dataset_id}.user_config`"
        logger.info(f"Executing BQ query in list_users: {query}")
        query_job = self.bq_client.client.query(query)
        results = query_job.result()
        return [dict(row) for row in results]