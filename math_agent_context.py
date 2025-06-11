"""
Math Agent with Shared Context

This module implements a multi-agent system for performing mathematical operations on arrays.
Agents share a common context to maintain state between operations.

Key Components:
- generator_agent: Creates arrays of random numbers
- math_agent: Performs mathematical operations on arrays
- assistant_agent: Coordinates between specialized agents
"""

import asyncio
import json
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

from openai import AsyncOpenAI
from agents import ItemHelpers, MessageOutputItem
from agents import Agent, AgentHooks, RunContextWrapper, Runner, function_tool
from agents import OpenAIChatCompletionsModel, TContext, set_tracing_disabled
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Configuration constants for the AI model
MODEL_NAME = "meta/llama-4-scout-17b-16e-instruct"
# Note: In production, use environment variables or secure storage for API keys
API_KEY = "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"
BASE_URL = "https://integrate.api.nvidia.com/v1"

# Disable tracing for cleaner output in this example
set_tracing_disabled(disabled=True)

# Initialize the OpenAI client with provided credentials
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

@dataclass
class SharedContext:
    """
    Shared context class that holds data accessible by all agents.
    
    Attributes:
        data: Optional list of floating-point numbers representing the current array
              being operated on by the agents.
    """
    data: Optional[List[float]] = None

@function_tool
async def exec_generate_data(wrapper: RunContextWrapper[SharedContext], code: str) -> str:
    """
    Execute Python code to generate data and store it in the shared context.
    
    Args:
        wrapper: Run context wrapper containing the shared context
        code: Python code string that generates a 'result' variable
        
    Returns:
        str: String representation of the generated data or empty list on error
        
    Note:
        The code must define a 'result' variable which will be stored in the shared context.
    """
    local_vars = {}
    try:
        # Execute the code in a controlled environment
        exec(code, globals(), local_vars)
        result = local_vars['result']
        
        # Convert numpy arrays to Python lists for JSON serialization
        if isinstance(result, np.ndarray):
            result = result.astype(float).tolist()
            
        # Update shared context
        wrapper.context.data = result
        return str(result)
    except Exception as e:
        print(f'ERROR: {e}')
        return str([])

@function_tool
async def array_operators(wrapper: RunContextWrapper[SharedContext], 
                         operator: str, 
                         factor: float = 0.0) -> str:
    """
    Apply a mathematical operation to the current array in the shared context.
    
    Args:
        wrapper: Run context wrapper containing the shared context
        operator: The operation to perform. Supported values:
                 - 'sum', 'max', 'min': Aggregate operations
                 - '+', 'plus', 'add': Addition
                 - '-', 'minus': Subtraction
                 - '*', 'multiply': Multiplication
                 - '/', 'divide': Division
        factor: The operand for arithmetic operations
        
    Returns:
        str: String representation of the operation result
        
    Note:
        Modifies the shared context in place with the operation result.
    """
    data = wrapper.context.data or []
    print(f"[DEBUG] Original data: {data} operator: {operator} factor: {factor}")
    values = np.array(data)

    # Apply the requested operation
    if operator in ['sum']:
        values = np.array([np.sum(values, axis=0)])
    elif operator in ['max']:
        values = np.array([np.max(values, axis=0)])
    elif operator in ['min']:
        values = np.array([np.min(values, axis=0)])
    elif operator in ['+', 'plus', 'add']:
        values = values + factor
    elif operator in ['-', 'minus']:
        values = values - factor
    elif operator in ['*', 'multiply']:
        values = values * factor
    elif operator in ['/', 'divide']:
        values = values / factor

    # Update shared context and return result
    result = values.astype(float).tolist()
    wrapper.context.data = result
    return str(result)


async def main():
    """
    Main async function that initializes agents and runs the interactive loop.
    
    The function:
    1. Creates a shared context for agent communication
    2. Initializes all agents with their specific roles and tools
    3. Starts an interactive loop to process user input
    """
    # Initialize the shared context that will be used by all agents
    shared_context = SharedContext()

    generator_agent = Agent[SharedContext](  
        name="generator_agent",
        instructions=prompt_with_handoff_instructions(
            """You are an agent that generates an array with N numbers and execute the code with provided tools.

            "Example: "generate 8 random numbers between 1 and 10" ->
            ```python
            result = np.random.randint(1, 10, size=8).astype(float)
            ```

            "Example: "generate 12 random numbers between 5 and 30" ->
            ```python
            result = np.random.randint(5, 30, size=12).astype(float)
            ```

            "Example: "generate 14 numbers between 15 and 80" ->
            ```python
            result = np.random.randint(15, 80, size=14).astype(float)
            ```
            """
        ),
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=[exec_generate_data],
    )

    math_agent = Agent[SharedContext](  
        name="math_agent",
        instructions=prompt_with_handoff_instructions(
            """You are a math operator agent.

            You must ONLY call the `array_operators` tool to apply arithmetic operations to an array.

            ❌ DO NOT respond in natural language.
            ✅ ALWAYS extract the correct operator and factor and call the tool.

            ### Operation Mapping Rules:
            - "compute sum of array" → operator="sum", factor=0
            - "get minimum value of array" → operator="min", factor=0
            - "get maximum value of array" → operator="max", factor=0
            - "add N", "plus N", "increase by N" → operator="+", factor=N
            - "subtract N", "minus N", "decrease by N" → operator="-", factor=N
            - "multiply by N", "times N" → operator="*", factor=N
            - "divide by N", "split by N" → operator="/", factor=N

            ### Examples:
            - Input: "Compute sum" → Call: array_operators(operator="sum", factor=0)
            - Input: "Find largest value" → Call: array_operators(operator="max", factor=0)
            - Input: "Get minimum value" → Call: array_operators(operator="min", factor=0)
            - Input: "add 2" → Call: array_operators(operator="+", factor=2)
            - Input: "minus 5" → Call: array_operators(operator="-", factor=5)
            - Input: "add 2" → Call: array_operators(operator="+", factor=2)
            - Input: "minus 5" → Call: array_operators(operator="-", factor=5)
            - Input: "multiply by 3" → Call: array_operators(operator="*", factor=3)
            - Input: "divide by 4" → Call: array_operators(operator="/", factor=4)
            - Input: "increase by 10" → Call: array_operators(operator="+", factor=10)
            - Input: "decrease by 6" → Call: array_operators(operator="-", factor=6)

            You MUST identify the correct operator and numerical factor and call the tool accordingly.
            """
        ),
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=[array_operators],
    )

    assistant_agent = Agent[SharedContext](  
        name="Assistant",
        instructions=prompt_with_handoff_instructions(
            """You are a math assistant responsible for coordinating and delegating tasks to specialized agents.

            DO NOT answer in text ALWAYS handoff to specialized agent.

            Handoff Rules:
            - Use `generator_agent` to generate list of numbers
            - Use `math_agent` to apply math operations to array
            """
        ),
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        handoffs=[generator_agent, math_agent],
    )

    # Main interactive loop
    print("Math Agent System Ready!")
    print("Type your command or press Ctrl+C to exit\n")
    
    while True:
        try:
            # Get user input
            user_input = input('User: ').strip()
            if not user_input:
                continue
                
            # Process the input through the agent system
            _ = await Runner.run(
                starting_agent=assistant_agent,
                input=user_input,
                context=shared_context,
            )

            # Display the current state of the data
            print(f'Assistant: {shared_context.data}\n')
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
