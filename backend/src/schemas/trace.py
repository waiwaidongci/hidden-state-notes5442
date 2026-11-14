from pydantic import BaseModel, Field, ConfigDict
from src.services.engine import MAX_TRACE_TOKENS


class TraceInput(BaseModel):
 model_config = ConfigDict(str_strip_whitespace=True)
 reference: str = Field(min_length=1, max_length=100)
 tokens: list[str] = Field(min_length=1, max_length=MAX_TRACE_TOKENS)


class TracePatch(BaseModel):
 tokens: list[str] = Field(min_length=1, max_length=MAX_TRACE_TOKENS)
