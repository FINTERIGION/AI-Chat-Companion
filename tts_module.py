import pyaudio
import base64
import threading
import dashscope
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat

class TTSManager:
    """
    Handles streaming TTS synthesis and audio playback with interruption support.
    """
    def __init__(self, api_key: str, model: str = "qwen3-tts-vd-realtime-2026-01-15"):
        dashscope.api_key = api_key
        self.model = model
        self.voice_id = None
        
        self._player = pyaudio.PyAudio()
        self._stream = None
        self._stop_event = threading.Event()
        self._is_playing = False
        self._stream_lock = threading.Lock()
        
        self.qwen_tts = None
        self.callback = None

    class TTSCallback(QwenTtsRealtimeCallback):
        def __init__(self, manager):
            self.manager = manager
            self.complete_event = threading.Event()

        def on_open(self) -> None:
            print('[TTS] Connection established')

        def on_close(self, close_status_code, close_msg) -> None:
            print(f'[TTS] Connection closed code={close_status_code}, msg={close_msg}')
            self.complete_event.set()

        def on_event(self, response: dict) -> None:
            try:
                event_type = response.get('type', '')
                if event_type == 'response.audio.delta':
                    if not self.manager._stop_event.is_set():
                        audio_data = base64.b64decode(response['delta'])
                        self.manager._write_to_stream(audio_data)
                elif event_type == 'session.finished':
                    self.complete_event.set()
            except Exception as e:
                print(f'[TTS Error] Callback exception: {e}')

    def _init_stream(self):
        with self._stream_lock:
            if self._stream is None or not self._stream.is_active():
                self._stream = self._player.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True
                )

    def _write_to_stream(self, data):
        with self._stream_lock:
            if self._stream and not self._stop_event.is_set():
                try:
                    self._stream.write(data)
                except Exception as e:
                    print(f"[TTS Error] Stream write failed: {e}")

    def synthesize_stream(self, text_generator, voice_id: str = None):
        """
        Consumes a text stream and plays audio in real-time.
        """
        self.interrupt() # Stop any ongoing playback
        self._stop_event.clear()
        self._init_stream()
        
        if voice_id:
            self.voice_id = voice_id

        self.callback = self.TTSCallback(self)
        self.qwen_tts = QwenTtsRealtime(
            model=self.model,
            callback=self.callback,
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime'
        )
        
        self.qwen_tts.connect()
        self.qwen_tts.update_session(
            voice=self.voice_id or "qwen-tts-vd-girlfriend-voice-20260223203142230-6aa3",
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode='server_commit'
        )

        self._is_playing = True
        try:
            for text_chunk in text_generator:
                if self._stop_event.is_set():
                    break
                if text_chunk:
                    self.qwen_tts.append_text(text_chunk)
            
            if not self._stop_event.is_set():
                self.qwen_tts.finish()
                self.callback.complete_event.wait()
        finally:
            self._is_playing = False
            if self.qwen_tts:
                self.qwen_tts.close()

    def interrupt(self):
        """
        Immediately stops audio playback and synthesis.
        """
        self._stop_event.set()
        with self._stream_lock:
            if self._stream:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception as e:
                    print(f"[TTS Error] Stream close failed: {e}")
                finally:
                    self._stream = None
        if self.qwen_tts:
            # Note: QwenTtsRealtime might not have a direct 'abort', 
            # but closing the connection and setting stop_event handles it.
            pass
        self._is_playing = False

    def terminate(self):
        self.interrupt()
        self._player.terminate()
