from pydantic import BaseModel
from typing import Union, Dict, Any, List, Optional

class InputSchema(BaseModel):
    prompt: str

class AgentPromptVariables(BaseModel):
    environment_name: str
    environment_info: Any
    short_term_memory: Optional[List[Dict[str, Any]]] = None
    long_term_memory: Optional[List[Dict[str, Any]]] = None
    perception: Optional[Any] = None
    observation: Optional[Any] = None
    action_space: Dict[str, Any] = Field(default_factory=dict)
    last_action: Optional[Any] = None
    reward: Optional[float] = None
    previous_strategy: Optional[str] = None