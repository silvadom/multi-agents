# Math Agent with Shared Context

A Python-based multi-agent system that performs mathematical operations on arrays using specialized AI agents. This project demonstrates agent collaboration through shared context and tool-based interactions.

## Features

- **Multi-Agent Architecture**: Three specialized agents working together:
  - `generator_agent`: Creates arrays of random numbers
  - `math_agent`: Performs mathematical operations on arrays
  - `assistant_agent`: Coordinates between the specialized agents
- **Shared Context**: Agents share and modify a common data structure
- **Interactive CLI**: Simple command-line interface for user interaction
- **Extensible Design**: Easy to add new operations or agents

## Prerequisites

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - numpy
  - openai
  - python-dotenv
  - pydantic
  - openai-agents (custom package)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in a `.env` file (if needed)

## Usage

Run the application:
```bash
python math_agent_context.py
```

### Example Commands
- Generate numbers: "generate 10 random numbers between 1 and 100"
- Perform operations:
  - "add 5"
  - "multiply by 3"
  - "compute sum"
  - "find maximum value"
  - "divide by 2"

## Project Structure

- `math_agent_context.py`: Main application file containing agent implementations
- `requirements.txt`: Project dependencies
- `README.md`: This documentation file

## License

This project is licensed under the MIT License - see the LICENSE file for details.
