from abc import ABC
from typing import Dict, List
from utils import load_config


class ChatResponse(ABC):
    """
    Represents a response from a chat model.

    This class encapsulates the content of a response from a chat model
    along with information about token usage.

    Attributes:
        content: The text content of the response.
        total_tokens: The total number of tokens used in the request and response.
    """

    def __init__(self, content: str, total_tokens: int) -> None:
        """
        Initialize a ChatResponse object.

        Args:
            content: The text content of the response.
            total_tokens: The total number of tokens used in the request and response.
        """
        self.content = content
        self.total_tokens = total_tokens

    def __repr__(self) -> str:
        """
        Return a string representation of the ChatResponse.

        Returns:
            A string representation of the ChatResponse object.
        """
        return f"ChatResponse(content={self.content}, total_tokens={self.total_tokens})"


class BaseLLM(ABC):
    """
    Abstract base class for language model implementations.

    This class defines the interface for language model implementations,
    including methods for chat-based interactions and parsing responses.
    """

    def __init__(self):
        """
        Initialize a BaseLLM object.
        """
        pass

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the language model and get a response.

        Args:
            messages: A list of message dictionaries, typically in the format
                     [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Returns:
            A ChatResponse object containing the model's response.
        """
        pass


class Anthropic(BaseLLM):
    """
    Anthropic language model implementation.

    This class provides an interface to interact with Anthropic's Claude language models
    through their API.

    Attributes:
        model (str): The Anthropic model identifier to use.
        max_tokens (int): The maximum number of tokens to generate in the response.
        client: The Anthropic client instance.
    """

    def __init__(self, model: str = "claude-3-7-sonnet-latest", max_tokens: int = 8192, **kwargs):
        """
        Initialize an Anthropic language model client.

        Args:
            model (str, optional): The model identifier to use. Defaults to "claude-3-7-sonnet-latest".
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.
            **kwargs: Additional keyword arguments to pass to the Anthropic client.
                - api_key: Anthropic API key. If not provided, uses config.yaml.
                - base_url: Anthropic API base URL. If not provided, uses the default Anthropic API endpoint.
        """
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        
        # Load config and get API key
        config = load_config()
        api_key = config.get('Anthropic', {}).get('api_key')
        if not api_key:
            raise ValueError("API key not found in config.yaml")
        
        base_url = None
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url, **kwargs)

    def chat(self, messages: List[Dict]) -> ChatResponse:
        """
        Send a chat message to the Anthropic model and get a response.

        Args:
            messages (List[Dict]): A list of message dictionaries, typically in the format
                                  [{"role": "system", "content": "..."},
                                   {"role": "user", "content": "..."}]

        Returns:
            ChatResponse: An object containing the model's response and token usage information.
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return ChatResponse(
            content=message.content[0].text,
            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
        )
