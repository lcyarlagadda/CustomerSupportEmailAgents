# TaskFlow Pro - AI Customer Support Automation

A production-ready multi-agent system for automating customer support email responses using LangGraph and RAG (Retrieval-Augmented Generation).

## Overview

This system automatically processes customer support emails through a sophisticated multi-agent workflow. It categorizes inquiries, retrieves relevant information from product documentation, generates personalized responses, and ensures quality before sending.

The system supports both local execution using HuggingFace Transformers and cloud-based inference via Groq API for faster processing.

## Features

- **Intelligent Classification** - Automatically categorizes emails into technical support, billing, product inquiries, feature requests, and feedback
- **RAG-Powered Responses** - Retrieves accurate information from product documentation to answer questions
- **Multi-Agent Workflow** - Four specialized agents work together: Classifier, RAG Agent, Response Generator, and QA Agent
- **Quality Assurance** - Built-in review system with automatic revision loop for sub-par responses
- **Flexible Deployment** - Supports local execution or cloud-based inference
- **Privacy First** - Customer data never leaves your infrastructure when using local models
- **Optimized Performance** - Includes INT8 quantization, parallel processing, and smart caching

## Architecture

The system uses LangGraph to orchestrate four AI agents:

1. **Email Classifier** - Categorizes incoming emails and assigns priority levels
2. **RAG Agent** - Searches product documentation and synthesizes relevant context
3. **Response Generator** - Drafts personalized email responses using category-specific templates
4. **QA Agent** - Reviews responses for quality, accuracy, and tone before sending

Each email flows through this pipeline with conditional routing based on classification and quality scores. Failed QA checks trigger automatic revision up to a configurable limit.

## Technology Stack

- **LangGraph** - Multi-agent workflow orchestration
- **LangChain** - AI framework and RAG implementation
- **HuggingFace Transformers** - Local LLMs (unlimited, private)
- **Groq API** - Fast cloud-based LLM inference (optional, 14.4k requests/day free tier)
- **Sentence Transformers** - Local embeddings
- **ChromaDB** - Vector database for document storage
- **Python 3.9+** - Core runtime

## Installation

### Prerequisites

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set HuggingFace Token (Optional)

For gated models like Llama:

1. Get token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Accept model license at the model's HuggingFace page
3. Create `.env` file:

```bash
HF_TOKEN=your_token_here
```

### Step 3: Initialize Vector Store

This creates embeddings from the product documentation:

```bash
python setup_vectorstore.py
```

First run will download the embedding model (~90MB). Subsequent runs use cached model.

## Quick Start

### Local Machine

Run the system in batch mode to process all current emails:

```bash
python main.py --mode batch --mock-emails
```

For continuous monitoring:

```bash
python main.py --mode continuous
```

### Using Groq API (Optional, Faster)

1. Get free API key: [console.groq.com/keys](https://console.groq.com/keys)
2. Add to `.env`:
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Provider: "groq" or "huggingface"
LLM_PROVIDER=huggingface

# HuggingFace Configuration
LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct
HF_TOKEN=your_token_here
USE_GPU=auto  # auto, true, false

# Groq Configuration (optional)
GROQ_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_key_here

# Email settings
GMAIL_EMAIL=support@taskflowpro.com
```

### Available Models

**HuggingFace Models:**
- `meta-llama/Llama-3.2-1B-Instruct` - 1B params, fast, good quality (default)
- `meta-llama/Llama-3.2-3B-Instruct` - 3B params, better quality
- `google/gemma-2-2b-it` - 2B params, alternative option
- `microsoft/phi-2` - 2.7B params, compact and efficient

**Groq Models:**
- `llama-3.1-8b-instant` - Fast, recommended default
- `llama-3.3-70b-versatile` - Higher quality
- `mixtral-8x7b-32768` - Excellent with 32k context

### Command Line Options

```bash
# Batch mode (process once and exit)
python main.py --mode batch --mock-emails

# Continuous monitoring
python main.py --mode continuous

# Adjust maximum revision attempts
python main.py --max-revisions 3 --mock-emails
```

## Project Structure

```
ProductSupportAgents/
├── agents/
│   ├── classifier.py          # Email classification agent
│   ├── rag_agent.py           # RAG retrieval agent
│   ├── response_generator.py  # Response drafting agent
│   └── qa_agent.py            # Quality assurance agent
├── workflows/
│   └── support_workflow.py    # LangGraph workflow orchestration
├── utils/
│   ├── config.py              # Configuration management
│   ├── email_handler.py       # Email operations
│   ├── vector_store.py        # Vector database management
│   ├── llm_loader.py          # HuggingFace LLM loader
│   ├── groq_loader.py         # Groq API loader
│   └── unified_llm_loader.py  # Unified LLM provider interface
├── data/
│   ├── product_docs/          # Product documentation (markdown)
│   └── test_emails/           # Test email JSON files
├── main.py                    # Main execution script
├── setup_vectorstore.py       # Vector store initialization
└── requirements.txt           # Python dependencies
```

## How It Works

### Email Processing Flow

1. **Receive** - System monitors inbox for new emails (or uses mock emails for testing)
2. **Classify** - Email categorized and prioritized
3. **Retrieve** - Relevant documentation fetched via RAG
4. **Generate** - Personalized response drafted with standardized format
5. **Review** - Quality check with automatic revision if needed
6. **Send** - Approved response delivered to customer

### Supported Categories

- **Technical Support** - Login issues, bugs, errors (uses documentation or escalates)
- **Product Inquiry** - Feature questions, how-to guides (answers from documentation)
- **Billing** - Subscriptions, payments, invoicing (always escalated)
- **Feature Request** - Enhancement suggestions (saved to file, acknowledgment sent)
- **Feedback** - General comments and reviews (saved to file, acknowledgment sent)
- **Unrelated** - Spam or off-topic (auto-filtered)

### Response Format

All responses follow a standardized format:

1. Salutation
2. Acknowledgment (apologize for issues, appreciate feedback)
3. Main content (solution, information, or escalation notice)
4. Closing: "Hope you have a great day!"
5. Signature: TaskFlow Pro Team

### Quality Assurance

Each response receives a quality score (0-10):

- **8.0-10.0** - Automatically approved and sent
- **7.0-7.9** - Approved with minor notes
- **Below 7.0** - Triggers revision or manual review

Escalation responses are automatically approved as they appropriately route to specialists.

## Performance

### Performance Metrics

| Configuration | Speed | Throughput |
|---------------|-------|------------|
| CPU (local) | 10-12s | ~300 emails/hour |
| GPU (local, INT8) | 3-6s | ~600-1200 emails/hour |
| Groq API | 1-2s | ~1800-3600 emails/hour |
| Cached responses | <1s | Instant |

### Optimizations Included

- **INT8 Quantization** - 2-3x faster on GPU (auto-enabled)
- **Parallel Processing** - RAG retrieval runs concurrently
- **Smart Caching** - Similar emails answered instantly
- **Token Optimization** - Agents use optimized token limits
- **Prompt Optimization** - Concise, focused prompts

### Benchmark Your System

```bash
python benchmark_performance.py
```

## Customization

### Add Your Product Documentation

1. Place markdown files in `data/product_docs/`
2. Rebuild vector store: `python setup_vectorstore.py`

### Modify Response Templates

Edit category-specific prompts in `agents/response_generator.py`

### Adjust Email Categories

Update `EMAIL_CATEGORIES` dictionary in `utils/config.py`

### Change LLM Model

Update `.env` file with desired model name.

## Testing

Test individual components:

```bash
# Test classifier
python agents/classifier.py

# Test RAG retrieval
python agents/rag_agent.py

# Test response generator
python agents/response_generator.py

# Test QA agent
python agents/qa_agent.py

# Test complete workflow
python workflows/support_workflow.py
```

## Production Deployment

### Connecting Real Email

The system uses mock emails by default. To connect Gmail:

1. Enable Gmail API in Google Cloud Console
2. Download `credentials.json` to project root
3. Run system - first run will prompt for authentication
4. Remove `--mock-emails` flag from command

### Scaling Considerations

- Run on dedicated server with adequate RAM
- Use GPU for faster inference or Groq API for cloud speed
- Implement load balancing for high volume
- Add monitoring and alerting
- Set up log aggregation

### Security

- All data processed locally when using HuggingFace models
- No external API calls for LLM inference (unless using Groq)
- Suitable for sensitive customer data
- GDPR/HIPAA compliant when using local models

## System Requirements

### Minimum

- CPU: Any modern processor
- RAM: 8GB
- Storage: 5GB free
- GPU: Not required

### Recommended

- CPU: 4+ cores
- RAM: 16GB
- Storage: 10GB free
- GPU: NVIDIA GPU with 4GB+ VRAM (optional, for speed)

## Troubleshooting

### Model Loading Issues

1. Check `HF_TOKEN` is set (for gated models)
2. Accept model license on HuggingFace
3. Try a different model

### CUDA Out of Memory

Use a smaller model or force CPU:

```bash
export USE_GPU=false
```

### Slow Performance

- Close other applications
- Use smaller model
- Reduce `--max-revisions` to 1
- Consider using Groq API
- Enable GPU if available

### Vector Store Errors

Reinitialize the vector store:

```bash
python setup_vectorstore.py
```

## Privacy & Security

### Data Privacy

When using local HuggingFace models:
- All processing happens locally
- Customer emails never sent to external APIs
- LLM runs on your infrastructure
- Complete control over data

When using Groq API:
- Email content is sent to Groq's servers
- Review Groq's privacy policy for compliance requirements

### Compliance

Suitable for various compliance requirements when using local models:
- GDPR (EU data protection)
- HIPAA (healthcare data)
- SOC 2 requirements
- Financial services regulations

## Development

### Running Tests

```bash
# Validate configuration
python utils/config.py

# Test vector store
python utils/vector_store.py

# Test individual agents (see Testing section)
```

### Code Structure

- **Agents**: Independent, single-responsibility components
- **Workflows**: LangGraph orchestration with conditional routing
- **Utils**: Shared utilities and configuration
- **Data**: Product documentation and vector storage

## Limitations

- Mock email handler by default (requires Gmail API integration for production)
- English language only (extensible to other languages)
- Synchronous processing (can be parallelized)
- Local vector store (can be upgraded to cloud services)

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - AI application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [HuggingFace](https://huggingface.co) - LLM models and embeddings
- [PyTorch](https://pytorch.org) - Deep learning framework
- [ChromaDB](https://www.trychroma.com) - Vector database
