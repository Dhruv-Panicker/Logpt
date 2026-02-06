'Task file that will contain dict for context defintion for log summary templates'


LOG_CONTEXTS = {
    "openssh": {
        "description": "SSH authentication and connection logs",
        "common_events": [
            "Failed password attempts",
            "Invalid user authentication",
            "Brute force attack patterns",
            "Successful logins",
            "Connection closed events",
            "Reverse DNS lookup failures"
        ],
        "key_fields": ["timestamp", "source_ip", "username", "action"],
        "summary_focus": "Security threats, attack patterns, authentication anomalies"
    },
    
    "linux": {
        "description": "Linux system logs (auth, kernel, cron)",
        "common_events": [
            "PAM authentication",
            "User sessions",
            "Cron job execution",
            "Kernel messages",
            "Service start/stop"
        ],
        "key_fields": ["timestamp", "service", "user", "action", "result"],
        "summary_focus": "System health, security events, service status"
    },
    
    "apache": {
        "description": "Apache web server logs",
        "common_events": [
            "mod_jk worker errors",
            "Worker environment state changes",
            "Child process initialization",
            "Configuration loading",
            "Client access attempts"
        ],
        "key_fields": ["timestamp", "log_level", "component", "message"],
        "summary_focus": "Server health, worker process issues, error patterns"
    },
    
    "hadoop": {
        "description": "Hadoop MapReduce job execution logs",
        "common_events": [
            "Job submission and initialization",
            "Task state transitions",
            "Container allocation",
            "Resource scheduling",
            "Task completion"
        ],
        "key_fields": ["timestamp", "job_id", "task_id", "state", "resource_info"],
        "summary_focus": "Job progress, resource allocation, task failures, bottlenecks"
    },
    
    "mac": {
        "description": "macOS system logs",
        "common_events": [
            "Power management (sleep/wake)",
            "Network state changes",
            "Application errors",
            "Kernel messages",
            "Sandbox violations"
        ],
        "key_fields": ["timestamp", "process", "subsystem", "message"],
        "summary_focus": "System stability, network issues, application errors"
    },
    
    "hdfs": {
        "description": "HDFS distributed file system logs",
        "common_events": [
            "Block operations",
            "DataNode packet handling",
            "NameSystem block updates",
            "Replication events",
            "Block termination"
        ],
        "key_fields": ["timestamp", "component", "block_id", "node", "action"],
        "summary_focus": "Storage operations, replication status, block management"
    },
    
    "openstack": {
        "description": "OpenStack cloud infrastructure logs",
        "common_events": [
            "API requests",
            "VM lifecycle operations",
            "Resource allocation",
            "Service status",
            "Request timing"
        ],
        "key_fields": ["timestamp", "request_id", "service", "endpoint", "status", "duration"],
        "summary_focus": "API performance, VM operations, resource usage, slow requests"
    },
    
    "spark": {
        "description": "Apache Spark executor and job logs",
        "common_events": [
            "Executor initialization",
            "Security setup",
            "JVM configuration",
            "Signal handler registration",
            "Environment setup"
        ],
        "key_fields": ["timestamp", "executor_id", "component", "action"],
        "summary_focus": "Executor health, initialization issues, configuration problems"
    },
    
    "hpc": {
        "description": "High Performance Computing cluster logs",
        "common_events": [
            "Job scheduling and allocation",
            "Node state changes",
            "Resource utilization",
            "MPI communication events",
            "Batch job completion",
            "Queue management"
        ],
        "key_fields": ["timestamp", "job_id", "node_id", "user", "state", "resource_info"],
        "summary_focus": "Job scheduling, node health, resource allocation, compute failures"
    },
    
    "thunderbird": {
        "description": "Thunderbird supercomputer system logs",
        "common_events": [
            "Cron job execution",
            "PAM session management",
            "NTP time synchronization",
            "Ganglia monitoring alerts",
            "InfiniBand fabric management",
            "SSH connections"
        ],
        "key_fields": ["timestamp", "node", "service", "user", "action"],
        "summary_focus": "Cluster health, node synchronization, monitoring alerts, scheduled tasks"
    },
    
    "zookeeper": {
        "description": "Apache ZooKeeper distributed coordination logs",
        "common_events": [
            "Leader election",
            "Quorum connection management",
            "Session handling",
            "Worker thread lifecycle",
            "Connection requests",
            "Topology changes"
        ],
        "key_fields": ["timestamp", "log_level", "thread", "component", "node_id", "session_id"],
        "summary_focus": "Cluster coordination, connection stability, quorum health, session management"
    },
    
    "bgl": {
        "description": "Blue Gene/L supercomputer system logs",
        "common_events": [
            "RAS events (Reliability, Availability, Serviceability)",
            "Hardware errors",
            "Node failures",
            "Memory errors",
            "Network errors",
            "Application crashes"
        ],
        "key_fields": ["timestamp", "node_id", "severity", "component", "error_code", "message"],
        "summary_focus": "Hardware failures, system reliability, error patterns, node health"
    },
    
    "health": {
        "description": "Health application logs",
        "common_events": [
            "Application state changes",
            "User interactions",
            "Data synchronization",
            "Background task execution",
            "Error conditions"
        ],
        "key_fields": ["timestamp", "event_type", "user_id", "action", "status"],
        "summary_focus": "Application health, user activity patterns, sync issues, errors"
    },
    
    "prox": {
        "description": "Proxifier network proxy logs",
        "common_events": [
            "Connection open/close",
            "Proxy routing decisions",
            "Data transfer statistics",
            "Connection lifetime tracking",
            "Application proxy assignments"
        ],
        "key_fields": ["timestamp", "application", "target_host", "proxy", "bytes_sent", "bytes_received", "lifetime"],
        "summary_focus": "Network traffic patterns, proxy usage, connection anomalies, data transfer volumes"
    }
}

# Different query types for varied training data
QUERY_TYPES = {
    "root_cause": {
        "instruction": "What is the root cause of any issues or errors shown in these logs?",
        "system": "You are a systems engineer performing root cause analysis on log data."
    },
    "action_items": {
        "instruction": "List the top 3 action items that should be taken based on these logs.",
        "system": "You are an operations engineer providing actionable recommendations from log analysis."
    }
}

#Function that will get the context based on the log type and build the query with context to be sent to the model
def build_summary_query_prompt(log_type: str): 
    context = LOG_CONTEXTS.get(log_type)
    if not context: 
        raise ValueError(f"Unsupported log type: {log_type}")
    
    prompt = f""" You are analayzing {context.get("description")}

        The events in these logs include: 
        {context.get("common_events")} 
        Key fields to pay attention to are: {context.get("key_fields")}

        Focus your summary on: {context.get("summary_focus")}

        Provide a clear, concise summary that:
        1. Identifies the main events or issues
        2. Notes any patterns or anomalies
        3. Suggests actionable insights when relevant
        4. Uses plain language for non-technical readers"""
    
    return prompt

#Function to build varied query prompt based on log type and query type
def build_varied_query_prompt(log_type: str, query_type: str):
    context = LOG_CONTEXTS.get(log_type)
    if not context:
        raise ValueError(f"Unsupported log type: {log_type}")   
    
    query_config = QUERY_TYPES.get(query_type)
    if not query_config:
        raise ValueError(f"Unsupported query type: {query_type}")
    
    prompt = f"""You are analyzing {context.get("description")}.

    {query_config.get("instruction")}

    Focus your response on: {context.get("summary_focus")}
    Use the following context to inform your answer:
    - Key fields to pay attention to are: {context.get("key_fields")}   
    Provide a response that is:
    1. Accurate and relevant to the query
    2. Informed by the specific context of these logs
    3. Clear and concise, avoiding unnecessary technical jargon"""

    return prompt
    