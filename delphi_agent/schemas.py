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

class ReflectionSchema(BaseModel):
    reflection: str = Field(..., description="Reflection on the observation and actions")
    strategy_update: List[str] = Field(..., description="Updated strategies based on the reflection and previous strategy")
    self_reward: float = Field(..., description="Self-assigned reward between 0.0 and 1.0")

class PerceptionSchema(BaseModel):
    monologue: str = Field(..., description="Agent's internal monologue about the perceived environment situation")
    emotions: List[str] = Field(..., description="Agent's current emotions given environemnt state")
    attention: List[str] = Field(..., description="Agent's current attetion given environment state")
    strategy: List[str] = Field(..., description="Agent's strategies given the current environment situation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")