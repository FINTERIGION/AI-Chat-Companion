import logging
import base64
import threading
import pyaudio
import dashscope
from dashscope.audio.qwen_omni import OmniRealtimeCallback, OmniRealtimeConversation, MultiModality
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

logger = logging.getLogger(__name__)

class STTManager:
    """
    Handles microphone input and streaming STT using OmniRealtimeConversation.
    """
    def __init__(self, api_key=None):
        if api_key:
            dashscope.api_key = api_key
        
        self.pya = None
        self.mic_stream = None
        self.conversation = None
        self.is_listening = False
        self.callback = None
        self._thread = None

    class _STTCallback(OmniRealtimeCallback):
        def __init__(self, manager):
            self.manager = manager

        def on_open(self) -> None:
            logger.info("STT connection opened.")

        def on_close(self, close_status_code, close_msg) -> None:
            logger.info(f"STT connection closed: {close_status_code} - {close_msg}")

        def on_event(self, response: dict) -> None:
            event_type = response.get('type')
            if event_type == 'conversation.item.input_audio_transcription.completed':
                transcript = response.get('transcript', '')
                if self.manager.callback:
                    self.manager.callback(transcript, is_final=True)
            elif event_type == 'conversation.item.input_audio_transcription.text':
                # Partial result
                transcript = response.get('stash', '')
                if self.manager.callback:
                    self.manager.callback(transcript, is_final=False)
            elif event_type == 'error':
                logger.error(f"STT Error: {response}")

    def start_listening(self, callback):
        """
        Begins audio capture and streaming.
        :param callback: Function to call with (text, is_final)
        """
        if self.is_listening:
            return

        self.callback = callback
        self.is_listening = True
        
        # Initialize PyAudio
        self.pya = pyaudio.PyAudio()
        self.mic_stream = self.pya.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=3200
        )

        # Initialize Conversation
        self.conversation = OmniRealtimeConversation(
            model='qwen3-asr-flash-realtime',
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime',
            callback=self._STTCallback(self)
        )
        self.conversation.connect()

        # Update session for STT
        transcription_params = TranscriptionParams(
            language='zh',
            sample_rate=16000,
            input_audio_format="pcm"
        )
        self.conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params,
        )

        # Start streaming thread
        self._thread = threading.Thread(target=self._stream_audio, daemon=True)
        self._thread.start()
        logger.info("STT listening started.")

    def _stream_audio(self):
        try:
            while self.is_listening and self.mic_stream:
                audio_data = self.mic_stream.read(3200, exception_on_overflow=False)
                if audio_data:
                    audio_b64 = base64.b64encode(audio_data).decode('ascii')
                    self.conversation.append_audio(audio_b64)
        except Exception as e:
            logger.error(f"Error in audio streaming thread: {e}")
        finally:
            self.stop_listening()

    def stop_listening(self):
        """
        Stops capture and closes the session.
        """
        if not self.is_listening:
            return

        self.is_listening = False
        
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
        
        if self.pya:
            self.pya.terminate()
            self.pya = None

        if self.conversation:
            self.conversation.close()
            self.conversation = None
            
        logger.info("STT listening stopped.")
