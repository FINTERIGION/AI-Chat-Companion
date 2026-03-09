import os
import sys
import logging
import threading
import json

from stt_module import STTManager
from llm_module import LLMManager
from tts_module import TTSManager    
from voice_module import VoiceManager
from memory_module import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Main")

class AIChatCompanion:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("Error: DASHSCOPE_API_KEY environment variable not set.")
            sys.exit(1)  

        self.settings_path = "settings.json"
        self.settings = self.load_settings()

        # Initialize Managers
        self.stt = STTManager(api_key=self.api_key)
        self.llm = LLMManager(api_key=self.api_key, model='qwen3.5-flash')
        self.tts = TTSManager(api_key=self.api_key)
        self.voice = VoiceManager(api_key=self.api_key)
        self.memory = MemoryManager()

        # Apply settings
        if self.settings.get("system_prompt"):
            self.llm.system_prompt = self.settings["system_prompt"]
            self.llm.clear_history()
        
        self.current_voice_id = self.settings.get("voice_id")

        # Load previous summaries and inject into LLM
        from datetime import datetime
        past_context = self.memory.get_all_summaries_context()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        system_content = f"Current time is: {current_time}.\n"
        if past_context:
            logger.info("Loaded previous summaries.")
            system_content += past_context
            
        self.llm.history.append({
            "role": "system",
            "content": system_content
        })

        self.is_running = True
        self.processing_lock = threading.Lock()
        self.is_responding = False

    def load_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading settings: {e}")
        return {}

    def save_setting(self, key, value):
        self.settings[key] = value
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def handle_stt_result(self, text: str, is_final: bool):
        # Partial result - handle barge-in
        if not is_final and self.is_responding:
            logger.info("Barge-in detected! Interrupting current response.")
            self.interrupt_response()

        if not text.strip():
            return

        if is_final:
            print(f"\n[User]: {text}")
            self.memory.add_message("user", text)
            
            # Start response generation in a separate thread to avoid blocking STT
            threading.Thread(target=self.generate_and_play_response, args=(text,), daemon=True).start()

    def interrupt_response(self):
        self.llm.cancel_generation()
        self.tts.interrupt()
        self.is_responding = False

    def generate_and_play_response(self, prompt: str):
        with self.processing_lock:
            self.is_responding = True
            try:
                # STT should keep listening.
                print("[AI]: ", end="", flush=True)
                
                # Create a generator for LLM response
                text_stream = self.llm.generate_response_stream(prompt)
                
                # We need to capture the full response for memory
                full_response_captured = []
                
                def captured_stream():
                    for chunk in text_stream:
                        if chunk:
                            print(chunk, end="", flush=True)
                            full_response_captured.append(chunk)
                            yield chunk

                # Pipe LLM stream to TTS
                self.tts.synthesize_stream(captured_stream(), voice_id=self.current_voice_id)
                
                full_text = "".join(full_response_captured)
                if full_text:
                    self.memory.add_message("assistant", full_text)
                    # Auto-save memory after each response
                    self.memory.summarize_and_persist()
                print() # New line after response
                
            except Exception as e:
                logger.error(f"Error during response generation: {e}")
            finally:
                self.is_responding = False

    def run(self):
        print("--- AI Chat Companion CLI ---")
        print("Commands: /exit, /system [prompt], /voice-clone [path], /voice-design [prompt]")
        
        if not self.current_voice_id:
            print("\n[Warning]: voice_id not detected! Please use /voice-design or /voice-clone first.")

        # Start STT in background
        self.stt.start_listening(self.handle_stt_result)
        
        try:
            while self.is_running:
                try:
                    cmd = input("> ")
                    if cmd.startswith("/exit"):
                        self.shutdown()
                    elif cmd.startswith("/system "):
                        new_prompt = cmd[len("/system "):].strip()
                        self.llm.system_prompt = new_prompt
                        self.llm.clear_history() # Reset with new system prompt
                        self.save_setting("system_prompt", new_prompt)
                        print(f"System prompt updated to: {new_prompt}")
                    elif cmd.startswith("/voice-clone "):
                        path = cmd[len("/voice-clone "):].strip()
                        try:
                            print(f"Cloning voice from {path}...")
                            self.current_voice_id = self.voice.clone_voice(path)
                            self.save_setting("voice_id", self.current_voice_id)
                            print(f"Voice cloned successfully. ID: {self.current_voice_id}")
                        except Exception as e:
                            print(f"Voice cloning failed: {e}")
                    elif cmd.startswith("/voice-design "):
                        parts = cmd[len("/voice-design "):].strip().split("|")
                        prompt = parts[0].strip()
                        preview_text = parts[1].strip() if len(parts) > 1 else None
                        
                        # Simple language detection
                        is_english = all(ord(c) < 128 for c in prompt)
                        lang = "en" if is_english else "zh"
                        
                        if not preview_text:
                            preview_text = "Hello, this is my new voice." if is_english else "你好，这是我的新声音。"

                        try:
                            print(f"Designing voice with prompt: {prompt} (Language: {lang})...")
                            self.current_voice_id = self.voice.design_voice(prompt, preview_text=preview_text, language=lang)
                            self.save_setting("voice_id", self.current_voice_id)
                            print(f"Voice designed successfully. ID: {self.current_voice_id}")
                        except Exception as e:
                            print(f"Voice design failed: {e}")
                    elif cmd.strip() == "":
                        continue
                    else:
                        # Treat as text input if user prefers typing
                        self.handle_stt_result(cmd, is_final=True)
                except EOFError:
                    self.shutdown()
                except Exception as e:
                    logger.error(f"Unexpected error in main loop: {e}")
                    # Continue loop unless it's a fatal error
        except KeyboardInterrupt:
            self.shutdown()
        finally:
            if self.is_running:
                self.shutdown()

    def shutdown(self):
        print("\nShutting down...")
        self.is_running = False
        self.stt.stop_listening()
        self.interrupt_response()
        self.tts.terminate()
        
        # Summarize and save memory
        print("Summarizing and saving conversation context...")
        self.llm.summarize_context()
        
        # Extract the summary from LLM history
        summary = ""
        for msg in reversed(self.llm.history):
            if msg["role"] == "system" and "Summary of previous conversation:" in msg["content"]:
                summary = msg["content"].replace("Summary of previous conversation: ", "").strip()
                break
        
        if summary:
            logger.info(f"New summary generated: {summary}")
            self.memory.save_summary(summary)
        else:
            logger.warning("No summary was generated or found in history.")
        
        self.memory.summarize_and_persist()
        print("Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    companion = AIChatCompanion()
    companion.run()
