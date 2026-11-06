"""
Conversation Memory module.
Manages multi-turn conversation sessions and history.
"""

import time
import uuid
from typing import List, Dict, Any, Optional


class SessionMemory:
    """In-memory chat history manager supporting multi-session persistence."""

    def __init__(self, max_history_turns: int = 20):
        self.max_history_turns = max_history_turns
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if not session_id or session_id not in self.sessions:
            new_id = session_id or str(uuid.uuid4())
            self.sessions[new_id] = []
            return new_id
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Appends a message (user or assistant) to a session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        entry = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": time.time()
        }

        self.sessions[session_id].append(entry)

        # Truncate old history if exceeds limit (keep newest)
        if len(self.sessions[session_id]) > self.max_history_turns * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_turns * 2:]

        return entry

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns the full message list for the session."""
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        """Clears messages for the specified session."""
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def export_markdown(self, session_id: str) -> str:
        """Exports the conversation as clean Markdown text."""
        history = self.get_history(session_id)
        if not history:
            return "# Chat History\n\nNo messages in this session yet."

        md_lines = ["# RAG AI Assistant - Conversation History\n"]
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            md_lines.append(f"### {role}\n{msg['content']}\n")
            if msg.get("sources"):
                md_lines.append("**Sources Consulted:**")
                for s in msg["sources"]:
                    md_lines.append(f"- `{s.get('doc_name')}` (Chunk #{s.get('chunk_index', 0)}, Score: {s.get('score', 0)})")
                md_lines.append("")
        return "\n".join(md_lines)
