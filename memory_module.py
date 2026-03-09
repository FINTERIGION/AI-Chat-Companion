import json
import os
from datetime import datetime
from typing import List, Dict

class MemoryManager:
    """
    Handles persistence of conversation summaries and history.
    """
    def __init__(self, storage_dir: str = "memory"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = os.path.join(self.storage_dir, f"session_{self.session_id}.json")
        
        self.history: List[Dict[str, str]] = []
        self.summary: str = ""
        self.past_summaries: List[Dict[str, str]] = self._load_all_summaries()

    def add_message(self, role: str, content: str):
        """
        Appends a message to the current session.
        """
        timestamp = datetime.now().isoformat()
        self.history.append({"role": role, "content": content, "timestamp": timestamp})
        self._save_to_disk()

    def get_context(self) -> Dict[str, any]:
        """
        Returns recent history and relevant summaries.
        """
        return {
            "summary": self.summary,
            "history": self.history
        }

    def get_all_summaries_context(self) -> str:
        """
        Returns a formatted string of all past summaries.
        """
        if not self.past_summaries:
            return ""
        
        context = "Summary of previous conversations:\n"
        for mem in self.past_summaries:
            context += f"- [{mem['timestamp']}] {mem['summary']}\n"
        return context

    def save_summary(self, summary: str):
        """
        Writes the summary to the current session file.
        """
        self.summary = summary
        self._save_to_disk()

    def load_summary(self) -> str:
        """
        Reads the summary from the local file.
        (Kept for compatibility, but get_all_summaries_context is preferred)
        """
        return self.get_all_summaries_context()

    def summarize_and_persist(self):
        """
        Compresses history and saves to storage.
        """
        self._save_to_disk()

    def _load_all_summaries(self) -> List[Dict[str, str]]:
        """
        Loads all summaries from the memory directory.
        """
        summaries = []
        for filename in sorted(os.listdir(self.storage_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "summary" in data and data["summary"]:
                            timestamp = data.get("timestamp", filename.replace("session_", "").replace(".json", ""))
                            summaries.append({
                                "timestamp": timestamp,
                                "summary": data["summary"]
                            })
                except Exception as e:
                    print(f"Error loading memory from {filepath}: {e}")
        return summaries

    def _save_to_disk(self):
        """
        Handles JSON serialization and file I/O.
        """
        data = {
            "timestamp": self.session_id,
            "summary": self.summary,
            "history": self.history
        }
        try:
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving memory to {self.current_file}: {e}")

    def _load_from_disk(self):
        """
        Handles JSON deserialization and file I/O.
        """
        pass # Not needed for current session as it starts fresh
