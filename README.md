# Plotly Agent

An AI-powered agent for creating Plotly visualizations using natural language descriptions.

## Installation

```bash
pip install -e .
```

## Usage

```python
import pandas as pd
from plotly_agent import PlotlyAgent

# Create a sample dataframe
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 30, 40, 50]
})

# Initialize the agent
agent = PlotlyAgent()

# Set the dataframe
agent.set_df(df)

# Create a visualization
response = agent.process_message("Create a line plot of x vs y")
```

## Features

- Natural language interface for creating Plotly visualizations
- Automatic code generation and execution
- Support for various plot types and customization
- Integration with pandas DataFrames
- Error handling and debugging support

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- plotly
- langchain-core
- langchain
- langchain-openai
- pydantic

## License

MIT 