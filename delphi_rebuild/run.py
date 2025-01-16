#!/usr/bin/env python
from dotenv import load_dotenv
from typing import Dict
from market_agents.agents.market_agent_prompter import MarketAgentPromptManager, AgentPromptVariables
from naptha_sdk.environment import Environment
from naptha_sdk.memory import Memory
from naptha_sdk.schemas import AgentRunInput, EnvironmentRunInput, MemoryRunInput
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from delphi_rebuild.prompts.market_agent_prompts import PERCEPTION_PROMPT, ACTION_PROMPT, REFLECTION_PROMPT
from delphi_rebuild.schemas import InputSchema
from delphi_rebuild.utils import load_prompt

load_dotenv()

logger = get_logger(__name__)

# You can create your module as a class or function
class DelphiAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment
        self.chat_environment = Environment(environment_deployment=self.orchestrator_deployment.environment_deployments[0])
        self.cognitive_memory = Memory(memory_deployment=self.deployment.memory_deployments[0])
        self.system_prompt = SystemPromptSchema(role=self.deployment.config.system_prompt["role"])
        self.inference_provider = InferenceClient(self.deployment.node)
        self.latest_message_id = 0

    async def chat(self, module_run):
        environment_run_input = EnvironmentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={
                "func_name": "update_state", 
                "func_input_data": {
                    "run_id": self.latest_message_id,
                    "messages": messages,
                    "current_message": current_message
                }
            },
            deployment=self.deployment.environment_deployments[0].model_dump(),
        )
        environment_response = await self.chat_environment.call_environment_func(environment_run_input)

        agent_response = self.agent_loop(module_run)

    async def agent_loop(self, module_run):
        perception_response = self.perceive(module_run)
        
        self.act()
        reflect()

    async def perceive(self, module_run):
        environment_run_input = EnvironmentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "get_global_state"},
            deployment=self.deployment.environment_deployments[0].model_dump(),
        )
        environment_info = await self.groupchat_environment.call_environment_func(environment_run_input)

        recent_interactions = self.retrieve_recent_memories(module_run)

        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            short_term_memory=short_term_memories,
            long_term_memory=[episode.dict() for episode in ltm_episodes],
        )

        prompt = PERCEPTION_PROMPT.format(variables.model_dump())

        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": prompt}
        ]
        logger.info(f"Messages: {messages}")

        llm_response = await self.inference_provider.run_inference({"model": self.deployment.config.llm_config.model,
                                                                    "messages": messages,
                                                                    "temperature": self.deployment.config.llm_config.temperature,
                                                                    "max_tokens": self.deployment.config.llm_config.max_tokens})
        perception_response = llm_response
        store_response = self.store_memory("perception", json.dumps(perception_response))
        logger.info(f"Perception response: {perception_response}")
        return perception_response

    async def act(self):
        environment_run_input = EnvironmentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "get_global_state"},
            deployment=self.deployment.environment_deployments[0].model_dump(),
        )
        environment_info = await self.groupchat_environment.call_environment_func(environment_run_input)

        action_prompt = ACTION_PROMPT.format(variables.model_dump())

        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": action_prompt}
        ]
        logger.info(f"Messages: {messages}")

        llm_response = await self.inference_provider.run_inference({"model": self.deployment.config.llm_config.model,
                                                                    "messages": messages,
                                                                    "temperature": self.deployment.config.llm_config.temperature,
                                                                    "max_tokens": self.deployment.config.llm_config.max_tokens})
        action_response = llm_response
        store_response = self.store_memory("action", json.dumps(action_response))
        logger.info(f"Action response: {action_response}")
        return action_response

    async def reflect(self):
        reflection_prompt = REFLECTION_PROMPT.format(variables.model_dump())

    async def retrieve_recent_memories(self, module_run):
        logger.info(f"Retrieving recent memories...")
        user_id = module_run.consumer_id
        mem_run_input = MemoryRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func": "retrieve_recent_memories", "func_input_data": {"cognitive_step": "conversation", "user_id": user_id, "limit": 5}},
            deployment=self.deployment.memory_deployments[0].model_dump(),
        )
        # Retrieve recent conversation from CognitiveMemory
        recent_interactions = await self.cognitive_memory.run_module(mem_run_input)
        return recent_interactions

    async def store_memory(self, cognitive_step, content):
        logger.info(f"Storing memory...")
        user_id = module_run.consumer_id
        mem_run_input = MemoryRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func": "store_cognitive_item", "func_input_data": {"cognitive_step": cognitive_step, "content": content}},
            deployment=self.deployment.memory_deployments[0].model_dump(),
        )

        store_response = await self.cognitive_memory.run_module(mem_run_input)

        return store_response

# Default entrypoint when the module is executed
async def run(module_run: Dict):
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    delphi_agent = DelphiAgent(module_run)
    return await delphi_agent.chat(module_run)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("agent", "delphi_rebuild/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    input_params = {
        "prompt": "hi",
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = asyncio.run(run(module_run))

    print("Response: ", response)