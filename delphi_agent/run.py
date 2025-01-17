#!/usr/bin/env python
from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.environment import Environment
from naptha_sdk.memory import Memory
from naptha_sdk.schemas import AgentRunInput, EnvironmentRunInput, MemoryRunInput, MemoryObject
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from delphi_agent.prompts.market_agent_prompts import PERCEPTION_PROMPT, ACTION_PROMPT, REFLECTION_PROMPT
from delphi_agent.schemas import AgentPromptVariables, InputSchema, PerceptionSchema
from delphi_agent.utils import load_prompt

load_dotenv()

logger = get_logger(__name__)

# You can create your module as a class or function
class DelphiAgent:
    def __init__(self, deployment: AgentDeployment):
        self.deployment = deployment
        self.chat_environment = Environment(environment_deployment=self.orchestrator_deployment.environment_deployments[0])
        self.cognitive_memory = Memory(memory_deployment=self.deployment.memory_deployments[0])
        self.episodic_memory = Memory(memory_deployment=self.deployment.memory_deployments[1])
        self.system_prompt = SystemPromptSchema(role=self.deployment.config.system_prompt["role"])
        self.inference_provider = InferenceClient(self.deployment.node)
        self.latest_message_id = 0
        self.last_action = None
        self.last_observation = None

    async def chat(self, module_run):
        environment_run_input = EnvironmentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={
                "func_name": "update_state", 
                "func_input_data": {
                    "message_id": self.latest_message_id,
                    "messages": messages,
                }
            },
            deployment=self.deployment.environment_deployments[0].model_dump(),
        )
        environment_response = await self.chat_environment.call_environment_func(environment_run_input)

        agent_response = self.agent_loop(module_run)

    async def agent_loop(self, module_run):
        perception = self.perceive(module_run)
        action = self.act(perception)
        reflect()

    async def perceive(self, module_run):
        logger.info(f"Getting global state...")
        environment_run_input = EnvironmentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "get_global_state"},
            deployment=self.deployment.environment_deployments[0].model_dump(),
        )
        environment_info = await self.groupchat_environment.call_environment_func(environment_run_input)

        logger.info(f"Retrieving recent memories...")
        user_id = module_run.consumer_id
        mem_run_input = MemoryRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func": "retrieve_recent_memories", "func_input_data": {"cognitive_step": "conversation", "user_id": user_id, "limit": 5}},
            deployment=self.deployment.memory_deployments[0].model_dump(),
        )
        # Retrieve recent conversation from CognitiveMemory
        recent_interactions = await self.cognitive_memory.run_module(mem_run_input)

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
        logger.info(f"Formatted perception prompt: {messages}")

        logger.info(f"Running inference...")
        llm_response = await self.inference_provider.run_inference({"model": self.deployment.config.llm_config.model,
                                                                    "messages": messages,
                                                                    "temperature": self.deployment.config.llm_config.temperature,
                                                                    "max_tokens": self.deployment.config.llm_config.max_tokens,
                                                                    "response_format": PerceptionSchema.model_json_schema()})
        perception_response = llm_response

        perception_mem = MemoryObject(
            agent_id=self.id,
            cognitive_step="perception",
            metadata={
                "environment_name": environment_name,
                "environment_info": environment_info
            },
            content=json.dumps(perception_response),
            created_at=datetime.now(timezone.utc),
        )


        logger.info(f"Storing perception in cognitive memory...")
        user_id = module_run.consumer_id
        perception_run_input = MemoryRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func": "store_cognitive_item", "func_input_data": {"cognitive_step": "perception", "content": json.dumps(perception_mem)}},
            deployment=self.deployment.memory_deployments[0].model_dump(),
        )
        perception_response = await self.cognitive_memory.run_module(perception_run_input)

        logger.info(f"Perception response: {perception_response}")
        return perception_response

    async def act(self):
        logger.info(f"Getting global state...")
        environment_run_input = EnvironmentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "get_global_state"},
            deployment=self.deployment.environment_deployments[0].model_dump(),
        )
        environment_info = await self.groupchat_environment.call_environment_func(environment_run_input)

        action_space = self.chat_environment.action_space
        serialized_action_space = {
            "allowed_actions": [action_type.__name__ for action_type in action_space.allowed_actions]
        }

        variables = AgentPromptVariables(
            environment_name=self.chat_environment.name,
            environment_info=environment_info,
            perception=perception,
            action_space=serialized_action_space,
            last_action=self.last_action,
            observation=self.last_observation
        )
        action_prompt = ACTION_PROMPT.format(variables.model_dump())
        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": action_prompt}
        ]
        logger.info(f"Formatted action prompt: {messages}")

        llm_response = await self.inference_provider.run_inference({"model": self.deployment.config.llm_config.model,
                                                                    "messages": messages,
                                                                    "temperature": self.deployment.config.llm_config.temperature,
                                                                    "max_tokens": self.deployment.config.llm_config.max_tokens,
                                                                    "response_format": LocalAction.model_json_schema()})
        action_response = llm_response

        action_mem = MemoryObject(
            agent_id=self.id,
            cognitive_step="action",
            metadata={
                "action_space": serialized_action_space,
                "last_action": self.last_action,
                "observation": self.last_observation,
                "perception": perception,
                "environment_name": environment_name,
                "environment_info": environment_info
            },
            content=json.dumps(action_response),
            created_at=datetime.now(timezone.utc),
        )

        logger.info(f"Storing action in cognitive memory...")
        user_id = module_run.consumer_id
        action_run_input = MemoryRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func": "store_cognitive_item", "func_input_data": {"cognitive_step": "action", "content": json.dumps(action_mem)}},
            deployment=self.deployment.memory_deployments[0].model_dump(),
        )
        action_response = await self.cognitive_memory.run_module(action_run_input)

        logger.info(f"Action response: {action_response}")
        return action_response

    async def reflect(self):
        logger.info(f"Getting global state...")
        environment_run_input = EnvironmentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func_name": "get_global_state"},
            deployment=self.deployment.environment_deployments[0].model_dump(),
        )
        environment_info = await self.groupchat_environment.call_environment_func(environment_run_input)

        total_weight = environment_reward_weight + self_reward_weight
        if total_weight == 0:
            raise ValueError("Sum of weights must not be zero.")

        environment_reward_weight /= total_weight
        self_reward_weight /= total_weight

        environment = self.environments[environment_name]
        last_step = environment.history.steps[-1][1] if environment.history.steps else None

        if last_step:
            reward = last_step.info.get('agent_rewards', {}).get(self.id, 0.0) or 0.0
            local_observation = last_step.global_observation.observations.get(self.id)
            observation = local_observation.observation if local_observation else {}
        else:
            observation = {}
            reward = 0.0

        logger.info(f"Storing observation in cognitive memory...")
        store_response = self.store_memory("observation", json.dumps(action_response))


        previous_strategy = "No previous strategy available"
        previous_reflection = await self.retrieve_recent_memories(cognitive_step='reflection', limit=1)
        if previous_reflection:
            last_reflection_obj = previous_reflection[0]
            previous_strategy = last_reflection_obj.metadata.get("strategy_update", "")
            if isinstance(previous_strategy, list):
                previous_strategy = " ".join(previous_strategy)

        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            observation=observation,
            last_action=self.last_action,
            reward=reward,
            previous_strategy=previous_strategy
        )

        reflection_prompt = REFLECTION_PROMPT.format(variables.model_dump())
        messages = [
            {"role": "system", "content": self.system_prompt.role},
            {"role": "user", "content": reflection_prompt}
        ]
        logger.info(f"Formatted reflection prompt: {messages}")

        llm_response = await self.inference_provider.run_inference({"model": self.deployment.config.llm_config.model,
                                                                    "messages": messages,
                                                                    "temperature": self.deployment.config.llm_config.temperature,
                                                                    "max_tokens": self.deployment.config.llm_config.max_tokens,
                                                                    "response_format": ReflectionSchema.model_json_schema()})
        reflection_response = llm_response

        reflection_mem = MemoryObject(
            agent_id=self.id,
            cognitive_step="reflection",
            metadata={
                "total_reward": round(total_reward_val, 4),
                "self_reward": round(self_reward, 4),
                "observation": observation,
                "strategy_update": response.get("strategy_update", ""),
                "environment_reward": round(environment_reward, 4),
                "environment_name": environment_name,
                "environment_info": environment_info
            },
            content=json.dumps(reflection_response.get("reflection", "")),
            created_at=datetime.now(timezone.utc),
        )

        logger.info(f"Storing reflection in cognitive memory...")
        user_id = module_run.consumer_id
        cogmem_run_input = MemoryRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func": "store_cognitive_item", "func_input_data": {"cognitive_step": "reflection", "content": json.dumps(reflection_mem)}},
            deployment=self.deployment.memory_deployments[0].model_dump(),
        )
        cogmem_response = await self.cognitive_memory.run_module(cogmem_run_input)

        logger.info(f"Storing in episodic memory...")
        user_id = module_run.consumer_id
        epmem_run_input = MemoryRunInput(
            consumer_id=module_run.consumer_id,
            inputs={"func": "store_episode", 
                    "func_input_data": 
                        {
                            "agent_id": self.id,
                            "task_query": query_str,
                            "steps": self.episode_steps,
                            "total_reward": round(total_reward_val),
                            "strategy_update": response.get("strategy_update", ""),
                            "metadata": None
                        }},
            deployment=self.deployment.memory_deployments[0].model_dump(),
        )
        epmem_response = await self.episodic_memory.run_module(epmem_run_input)

        logger.info(f"Reflection response: {reflection_response}")
        return reflection_response

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

    deployment = asyncio.run(setup_module_deployment("agent", "delphi_agent/configs/deployment.json", node_url = os.getenv("NODE_URL")))

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