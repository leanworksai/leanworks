# LeanWorks

Internal Python package for LeanWorks AI.

## Installation

From GitHub:
```bash
pip install git+https://github.com/yourusername/leanworks.git
```

For development:
```bash
git clone https://github.com/yourusername/leanworks.git
cd leanworks
pip install -e .
```

## Features

- RAG (Retrieval Augmented Generation)
- Cloud storage integration
- Secret management

## Usage

```python
from leanworks import rag
from leanworks.rag import Chat

# Basic RAG example
retriever = rag.get_retriever()
results = retriever.query("your query here")
```

## Requirements

- Python 3.10 or higher

## License

[Your chosen license] 