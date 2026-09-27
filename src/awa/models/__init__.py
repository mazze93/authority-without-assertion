"""One protocol, two adapters. Deliberately not a provider framework."""
from .base import ModelClient, ModelRequest, ModelResponse, Transport, http_post_json
from .openai_compatible import OpenAICompatibleClient
from .openai_responses import OpenAIResponsesClient

__all__ = ["ModelClient", "ModelRequest", "ModelResponse", "Transport", "http_post_json",
           "OpenAICompatibleClient", "OpenAIResponsesClient"]
