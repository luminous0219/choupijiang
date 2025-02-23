import json
import uuid
from datetime import datetime
from pathlib import Path

class SessionManager:
    def __init__(self):
        self.sessions_dir = Path("data/sessions")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> str:
        """Create new empty session and return ID"""
        session_id = str(uuid.uuid4())
        self.save_session(session_id, {
            "created_at": datetime.now().isoformat(),
            "history": []
        })
        return session_id
    
    def save_session(self, session_id: str, data: dict):
        """Save session data to file"""
        with open(self.sessions_dir / f"{session_id}.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def load_session(self, session_id: str) -> dict:
        """Load session data or create new if not found"""
        try:
            with open(self.sessions_dir / f"{session_id}.json") as f:
                data = json.load(f)
                # Ensure history exists for backward compatibility
                if "history" not in data:
                    data["history"] = []
                return data
        except FileNotFoundError:
            return self.create_session()
    
    def clear_session(self, session_id: str):
        """Reset session while keeping ID"""
        self.save_session(session_id, {
            "created_at": datetime.now().isoformat(),
            "history": [],
            "context": {}  # Add context storage
        })

    def update_context(self, session_id: str, context: dict):
        """Update session context with new data"""
        session = self.load_session(session_id)
        if "context" not in session:
            session["context"] = {}
        session["context"].update(context)
        self.save_session(session_id, session)
