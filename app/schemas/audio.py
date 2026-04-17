from typing import List, Optional
from pydantic import BaseModel


class Segment(BaseModel):
    id: str
    type: str = "transcript.text.segment"
    text: str
    start: float
    end: float
    speaker: Optional[str] = None


class InputTokenDetails(BaseModel):
    text_tokens: int = 0
    audio_tokens: int


class Usage(BaseModel):
    type: str = "tokens"
    input_tokens: int
    input_token_details: InputTokenDetails
    output_tokens: int
    total_tokens: int


class AudioTranscription(BaseModel):
    text: str
    segments: Optional[List[Segment]] = None
    usage: Usage
