PERCEPTION_PROMPT = """Perceive the current state of the {environment_name} environment:

## Environment State
{environment_info}

## Recent Memories
{short_term_memory}

## Long Term Memories
{long_term_memory}

Generate a brief monologue about your current perception of this environment.
Use emojis to fully express yourself when needed.
"""

ACTION_PROMPT = """Generate an action for the {environment_name} environment based on the following:

## Current Perception
{perception}

## Environment State
{environment_info}

## Last Observation
{observation}

## Available Actions
{action_space}

Choose an appropriate action for this environment.
"""

REFLECTION_PROMPT = """Reflect on this observation from the {environment_name} environment:

## Current Observation
{observation}

## Environment State
{environment_info}

## Last Action
{last_action}

## Reward
{reward}

Actions:
1. Reflect on your actions, observations and reward (if available)
2. Assign yourself a reward between 0.0 and 1.0 based on your performance and alignment with your objectives & strategy.
3. Update strategy based on this reflection and your previous strategy.

## Previous Strategy
{previous_strategy}
"""
