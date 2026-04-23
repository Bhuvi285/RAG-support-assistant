# hitl.py
import json
import os
from datetime import datetime

ESCALATION_LOG = "escalation_log.json"

def log_escalation(query: str, reason: str):
    """Log escalated queries for human agents."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "reason": reason,
        "status": "pending"
    }

    log = []
    if os.path.exists(ESCALATION_LOG):
        with open(ESCALATION_LOG, "r") as f:
            log = json.load(f)

    log.append(entry)

    with open(ESCALATION_LOG, "w") as f:
        json.dump(log, f, indent=2)

    return entry

def get_pending_escalations():
    """Fetch all pending human review items."""
    if not os.path.exists(ESCALATION_LOG):
        return []
    with open(ESCALATION_LOG, "r") as f:
        log = json.load(f)
    return [e for e in log if e["status"] == "pending"]

def resolve_escalation(timestamp: str, human_reply: str):
    """Mark an escalation as resolved with human response."""
    if not os.path.exists(ESCALATION_LOG):
        return

    with open(ESCALATION_LOG, "r") as f:
        log = json.load(f)

    for entry in log:
        if entry["timestamp"] == timestamp:
            entry["status"] = "resolved"
            entry["human_reply"] = human_reply

    with open(ESCALATION_LOG, "w") as f:
        json.dump(log, f, indent=2)