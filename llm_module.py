import logging
import dashscope
from typing import List, Dict, Generator, Optional

# Set base API URL as required by some environments
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages conversation state and streaming chat completions using DashScope.
    Supports both text-only (Generation) and multimodal (MultiModalConversation) models.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = 'qwen3.5-flash'):
        """
        Initializes the LLMManager.
        :param api_key: DashScope API key. If None, uses dashscope.api_key or environment variable.
        :param model: The model name to use for completions.
        """
        if api_key:
            dashscope.api_key = api_key
        
        self.model = model
        self.history: List[Dict] = []
        self.is_generating = False
        self._cancel_requested = False
        
        # System prompt designed for oral communication
        self.system_prompt = (
            "You are a helpful, friendly AI assistant. "
            "Your responses should be short, concise, and suitable for oral communication. "
            "Avoid long lists or complex formatting. Speak naturally and keep it brief."
        )
        
        # Initialize history with system prompt
        self._reset_history()

    def _reset_history(self):
        """Resets the conversation history to the initial state with the system prompt."""
        # Use standard text format for history; convert to multimodal format only if needed
        self.history = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

    def generate_response_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Sends user prompt to LLM and yields text chunks as they arrive.
        Updates history with user prompt and assistant response.
        
        :param prompt: The user's input text.
        :yield: Text chunks from the LLM response.
        """
        self.is_generating = True
        self._cancel_requested = False
        
        # Add user message to history
        user_message = {
            "role": "user",
            "content": prompt
        }
        self.history.append(user_message)

        full_response = ""
        try:
            # Determine if we should use MultiModalConversation or Generation
            # qwen-max, qwen-plus, qwen-turbo are text models.
            # qwen-vl-*, qwen3.5-flash (based on example) are multimodal.
            is_multimodal = 'vl' in self.model.lower() or 'qwen3' in self.model.lower()
            
            if is_multimodal:
                # Convert history to multimodal format (list of dicts for content)
                mm_history = []
                for msg in self.history:
                    mm_history.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}]
                    })
                
                responses = dashscope.MultiModalConversation.call(
                    model=self.model,
                    messages=mm_history,
                    stream=True,
                    incremental_output=True,
                    enable_thinking=False
                )
            else:
                # Use standard Generation API for text models
                responses = dashscope.Generation.call(
                    model=self.model,
                    messages=self.history,
                    result_format='message',
                    stream=True,
                    incremental_output=True,
                    enable_thinking=False
                )

            for chunk in responses:
                if self._cancel_requested:
                    logger.info("LLM generation cancelled by user.")
                    break
                
                if chunk.status_code == 200:
                    # Extract text from the response chunk
                    choices = chunk.output.choices
                    if choices and choices[0].message.content:
                        content = choices[0].message.content
                        text_chunk = ""
                        
                        if isinstance(content, list):
                            # MultiModalConversation format
                            if len(content) > 0:
                                text_chunk = content[0].get('text', '')
                        else:
                            # Generation format (string)
                            text_chunk = content
                            
                        if text_chunk:
                            full_response += text_chunk
                            yield text_chunk
                else:
                    logger.error(f"LLM Error: {chunk.code} - {chunk.message}")
                    yield f"\n[Error: {chunk.message}]\n"
                    break

            # Add assistant response to history if any text was generated
            if full_response:
                self.history.append({
                    "role": "assistant",
                    "content": full_response
                })

        except Exception as e:
            logger.error(f"Exception in LLM generation: {e}")
            yield f"\n[Exception: {str(e)}]\n"
        finally:
            self.is_generating = False
            self._cancel_requested = False

    def cancel_generation(self):
        """Interrupts the current generation process."""
        if self.is_generating:
            self._cancel_requested = True

    def summarize_context(self):
        """
        Condenses the conversation history to maintain long-term memory.
        Replaces the current history with a summary of previous interactions.
        """
        # Only summarize if there's something to summarize beyond the system prompt
        if len(self.history) <= 1:
            return

        logger.info("Summarizing conversation context for long-term memory...")
        
        # Prepare a prompt for summarization
        summary_instruction = (
            "Please provide a very concise summary of the key points and user preferences "
            "from the conversation above. This summary will be used as context for future interactions."
        )
        
        # Create a temporary history for the summarization request
        temp_history = self.history + [
            {"role": "user", "content": summary_instruction}
        ]
        
        try:
            is_multimodal = 'vl' in self.model.lower() or 'qwen3' in self.model.lower()
            
            if is_multimodal:
                mm_history = []
                for msg in temp_history:
                    mm_history.append({
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}]
                    })
                response = dashscope.MultiModalConversation.call(
                    model=self.model,
                    messages=mm_history,
                    stream=False
                )
            else:
                response = dashscope.Generation.call(
                    model=self.model,
                    messages=temp_history,
                    result_format='message',
                    stream=False
                )
            
            if response.status_code == 200:
                choices = response.output.choices
                if choices and choices[0].message.content:
                    content = choices[0].message.content
                    summary_text = ""
                    if isinstance(content, list):
                        if len(content) > 0:
                            summary_text = content[0].get('text', '')
                    else:
                        summary_text = content
                        
                    if summary_text:
                        # Reset history and add the summary as a system context message
                        self._reset_history()
                        self.history.append({
                            "role": "system", 
                            "content": f"Summary of previous conversation: {summary_text}"
                        })
                        logger.info("Conversation history summarized and condensed.")
            else:
                logger.error(f"Summarization failed: {response.message}")
        except Exception as e:
            logger.error(f"Exception during summarization: {e}")

    def clear_history(self):
        """Clears the entire conversation history and resets to initial state."""
        self._reset_history()
        logger.info("Conversation history cleared.")
