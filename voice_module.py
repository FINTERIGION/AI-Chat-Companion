import requests
import base64
import pathlib

class VoiceManager:
    """
    Handles voice customization including cloning and design using DashScope API.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"

    def clone_voice(self, audio_path: str, preferred_name: str = "cloned_voice") -> str:
        """
        Creates a voice ID from an audio file (Voice Cloning).
        """
        file_path_obj = pathlib.Path(audio_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Determine MIME type based on extension
        ext = file_path_obj.suffix.lower()
        mime_type = "audio/mpeg" if ext == ".mp3" else "audio/wav"
        
        base64_str = base64.b64encode(file_path_obj.read_bytes()).decode()
        data_uri = f"data:{mime_type};base64,{base64_str}"

        payload = {
            "model": "qwen-voice-enrollment",
            "input": {
                "action": "create",
                "target_model": "qwen3-tts-vc-realtime-2026-01-15",
                "preferred_name": preferred_name,
                "audio": {"data": data_uri}
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.base_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Voice cloning failed: {response.status_code}, {response.text}")

        try:
            return response.json()["output"]["voice"]
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse voice cloning response: {e}")

    def design_voice(self, voice_prompt: str, preview_text: str = "你好，这是我的新声音。", preferred_name: str = "designed_voice", language: str = "zh") -> str:
        """
        Creates a voice ID from a text description (Voice Design).
        """
        payload = {
            "model": "qwen-voice-design",
            "input": {
                "action": "create",
                "target_model": "qwen3-tts-vd-realtime-2026-01-15",
                "voice_prompt": voice_prompt,
                "preview_text": preview_text,
                "preferred_name": preferred_name,
                "language": language
            },
            "parameters": {
                "sample_rate": 24000,
                "response_format": "wav"
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(self.base_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Voice design failed: {response.status_code}, {response.text}")

        try:
            result = response.json()
            voice_id = result["output"]["voice"]
            
            preview_audio_data = result["output"].get("preview_audio", {}).get("data")
            if preview_audio_data:
                voice_dir = pathlib.Path("voice")
                voice_dir.mkdir(exist_ok=True)
                
                if preview_audio_data.startswith("data:"):
                    base64_str = preview_audio_data.split(",", 1)[1]
                else:
                    base64_str = preview_audio_data
                    
                audio_bytes = base64.b64decode(base64_str)
                
                file_path = voice_dir / f"{voice_id}.wav"
                file_path.write_bytes(audio_bytes)
            
            return voice_id
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse voice design response: {e}")
