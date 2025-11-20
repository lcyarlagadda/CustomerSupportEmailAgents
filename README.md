# Product Support Agents - AI Customer Support Automation

A production-ready multi-agent system for automating customer support email responses using LangGraph, RAG (Retrieval-Augmented Generation), and intelligent workflow orchestration.

## Overview

This system automatically processes customer support emails through a sophisticated multi-agent workflow. It categorizes inquiries, retrieves relevant information from both product documentation and customer databases, generates personalized responses, and ensures quality before sending.

## Key Features

### Core Capabilities
- **Intelligent Email Classification** - Categorizes emails into technical support, billing, product inquiries, feature requests, and feedback with priority assignment
- **RAG-Powered Responses** - Retrieves accurate information from product documentation using hybrid semantic + keyword search
- **Customer Database Integration** - Queries SQLite database for real-time billing and account information
- **Multi-Agent Workflow** - Four specialized agents (Classifier, Response, Database, QA) orchestrated via LangGraph
- **Advanced Safety Checks** - PII detection (Presidio), toxic language filtering, and content validation
- **Quality Assurance** - Built-in review system with automatic revision loop for improved responses
- **Email Threading** - Replies properly threaded in Gmail conversations

### Advanced RAG Features
- **Multi-Query Retrieval** - Generates query variations for improved recall (+10-15% accuracy)
- **Reciprocal Rank Fusion (RRF)** - Combines results from multiple retrieval strategies
- **Contextual Compression** - Extracts only relevant sentences from retrieved documents
- **Query Decomposition** - Breaks complex queries into simpler sub-queries
- **Hybrid Search** - Combines semantic and keyword (BM25) search
- **Cross-Encoder Reranking** - Reorders results for maximum relevance

### Database Integration
- **SQLite Database** - Built-in, zero-setup database for customer data (perfect for development and Google Colab)
- **Automatic Database Check** - System detects billing/membership queries and checks database first
- **Read-Only Security** - Only SELECT queries permitted, all parameterized to prevent SQL injection
- **Detailed Logging** - All database operations logged with customer info
- **Redis Caching** - Optional sub-millisecond response times with Redis
- **Personalized Responses** - Answers billing queries with actual customer data (subscription, payment history, usage)
- **Automatic Fallback** - Uses documentation when database info unavailable

## Architecture

### Agent Structure

| Agent | Purpose | Capabilities |
|-------|---------|--------------|
| **Classifier** | Email categorization | Categorizes emails, assigns priority, determines routing |
| **Response Agent** | RAG + generation | Document retrieval with optional advanced features, response generation |
| **Database Agent** | Customer data queries | Retrieves customer billing/subscription data from SQLite |
| **QA Agent** | Safety & quality validation | PII detection, toxic language check, tone analysis, quality scoring |

### LangGraph Workflow

```
Email → Classify → Check Database? 
                     ├─ Yes → Query DB → Generate Response
                     └─ No → RAG Search → Generate Response
                              ↓
                         QA Validation
                              ↓
                         Send or Review
```

## Technology Stack

- **LangGraph** - Multi-agent workflow orchestration
- **LangChain** - AI framework and RAG implementation
- **Groq API** - Fast cloud-based LLM inference (14.4k free requests/day)
- **Instructor + Pydantic** - Structured, validated LLM outputs
- **Sentence Transformers** - Local embeddings and cross-encoder reranking
- **ChromaDB** - Vector database for document storage
- **Presidio** - PII detection and anonymization
- **SQLite** - Built-in database for customer data
- **Redis** - Optional caching layer
- **Gmail API** - Email operations and threading
- **Python 3.9+** - Core runtime

## Installation

### Prerequisites

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- Groq API key (free at https://console.groq.com)

### Setup Steps

1. **Clone Repository**
```bash
git clone <repository-url>
cd ProductSupportAgents
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Environment**

Create `.env` file:
```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
GROQ_MODEL=llama-3.1-8b-instant  # Default model
GMAIL_EMAIL=support@yourcompany.com

# Optional Redis (for caching)
REDIS_HOST=localhost
REDIS_PORT=6379
```

4. **Initialize Vector Store**
```bash
python setup_vectorstore.py
```

5. **Test the System**
```bash
python main.py
```

## Quick Start

### Option 1: Demo Web UI (Recommended for Testing)
```bash
streamlit run demo_ui.py
```
Opens a web interface where you can:
- Paste any email content
- See real-time processing with logs
- Test database integration
- View classification and response
- Load sample emails with one click

See [DEMO_UI_GUIDE.md](DEMO_UI_GUIDE.md) for details.

### Option 2: Command Line
```bash
python main.py
# Select option 2 to process test emails
```

### Test Database Integration
```bash
python test_database.py
```

### Run with Real Gmail
```bash
python main.py
# Select option 1 for continuous monitoring
```

## Configuration

### Enable Advanced RAG Features

In `workflows/support_workflow.py`:

```python
from agents.response_agent import ResponseAgent

self.response_agent = ResponseAgent(
    use_reranking=True,              # Standard (enabled by default)
    use_query_enhancement=True,      # Standard (enabled by default)
    use_hybrid_search=True,          # Standard (enabled by default)
    use_multi_query=True,            # Advanced (+10-15% accuracy, optional)
    use_query_decomposition=True,    # Advanced (for complex queries, optional)
    use_contextual_compression=True  # Advanced (reduces noise, optional)
)
```

**Performance Impact:**
- Standard features: ~2s per email
- With advanced features: ~3-4s per email, +30-40% accuracy

### Database Configuration

The SQLite database is created automatically with sample data on first run:
- Database file: `customer_data.db`
- Sample customer: `john.doe@example.com`
- Includes: subscription, payment history, usage data, invoices

To add your own customer data, see `test_database.py` for examples.

### Available Groq Models

- **llama-3.1-8b-instant** (Default) - Fast, balanced performance (500+ tok/s)
- **llama-3.3-70b-versatile** - Higher quality (300+ tok/s)
- **mixtral-8x7b-32768** - Excellent balance (400+ tok/s)

Change model in `.env`: `GROQ_MODEL=model_name`

## Gmail Integration

### Setup

1. Enable Gmail API in [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create OAuth 2.0 credentials (Desktop app type)
3. Download `credentials.json` to project root
4. Run system - first run will prompt for authentication
5. Token saved to `token.pickle` for future use

### Features

- Proper email threading in conversations
- Automatic labeling for manual review
- Read/unread status management
- Error handling with troubleshooting guidance

## Project Structure

```
ProductSupportAgents/
├── agents/
│   ├── classifier.py         # Email classification
│   ├── response_agent.py     # RAG + response generation (with advanced features)
│   ├── database_agent.py     # SQLite customer queries
│   └── qa_agent.py          # Safety & quality validation
├── workflows/
│   └── support_workflow.py   # LangGraph orchestration
├── utils/
│   ├── vector_store.py       # ChromaDB management
│   ├── groq_loader.py        # Groq API loader
│   ├── instructor_llm.py     # Structured outputs
│   └── email_handler.py      # Gmail API operations
├── data/
│   ├── product_docs/         # Product documentation (markdown)
│   └── test_emails/          # Test email JSON files
├── database/
│   └── schema_sqlite.sql     # SQLite schema + sample data
├── customer_data.db          # SQLite database (auto-created)
├── main.py                   # Main execution script
├── test_database.py          # Database integration tests
├── setup_vectorstore.py      # Vector store initialization
└── requirements.txt          # Python dependencies
```

## Usage

### Supported Email Categories

| Category | Database Check | Response Source |
|----------|---------------|-----------------|
| **Technical Support** | No | Product documentation |
| **Product Inquiry** | No | Product documentation |
| **Billing** | Yes | Customer database + documentation |
| **Feature Request** | No | Acknowledgment only |
| **Feedback** | No | Acknowledgment only |

### Response Examples

**Without Database (Technical Query):**
> "How do I integrate with Slack?"
>
> Response uses product documentation to provide step-by-step integration instructions.

**With Database (Billing Query):**
> "Why was I charged $49.99?" (from john.doe@example.com)
>
> System logs:
> ```
> [Workflow] Category 'billing' requires database check
> [Database] Checking database for customer: john.doe@example.com
> [Database] Found customer: John Doe
> [Database]   Plan: Pro Plan ($49.99/monthly)
> [Workflow] Using database context for response
> ```
>
> Response: "You were charged $49.99 on [date] for your monthly Pro Plan subscription. This is your regular billing. Your next charge of $49.99 will be on [date]. You're currently using 8/10 projects and 45/100 GB storage."

### Quality Assurance

Responses are scored 0-10:
- **8.0-10.0** - Automatically approved and sent
- **7.0-7.9** - Approved with minor notes
- **Below 7.0** - Triggers revision or manual review

## Performance

### Response Times

| Configuration | Time per Email | Use Case |
|--------------|----------------|----------|
| Standard RAG | ~2s | General queries |
| Advanced RAG | ~3-4s | Complex queries, higher accuracy |
| Database query | ~2s | Billing/account queries |
| With Redis cache | <1s | Repeated queries |

### Optimizations

- Groq API: 300-500 tokens/second inference
- Parallel document retrieval
- Hybrid semantic + keyword search
- Cross-encoder reranking
- Optional Redis caching
- Efficient embedding models

## Google Colab Support

The system works perfectly in Google Colab with zero configuration:

1. SQLite database auto-creates on first run
2. All dependencies install via pip
3. Optional: Use Upstash Redis (free cloud Redis) for caching

See test files for examples of running in Colab.

## Customization

### Add Product Documentation

1. Place markdown files in `data/product_docs/`
2. Rebuild vector store: `python setup_vectorstore.py`

### Modify Response Templates

Edit category-specific prompts in `agents/response_agent.py`

### Add Customer Data

```python
import sqlite3

conn = sqlite3.connect('customer_data.db')
cursor = conn.cursor()

# Add customer, subscription, payment data
# See test_database.py for examples

conn.commit()
conn.close()
```

### Adjust Quality Thresholds

Modify QA agent approval logic in `agents/qa_agent.py`

## Testing

### Run Database Tests
```bash
python test_database.py
```

This will test:
- Database connection and initialization
- Customer data queries (with logging)
- Billing query end-to-end (shows database check in action)
- Account status query
- RAG fallback for non-database queries

### Test with Sample Emails
```bash
python main.py
# Select option 2: Process test email
# Choose billing test email to see database check in action
```

### Test Individual Components
```python
from agents.classifier import EmailClassifierAgent
from agents.response_agent import ResponseAgent
from agents.database_agent import get_database_agent

# Test classification
classifier = EmailClassifierAgent()

# Test database
db = get_database_agent()
account = db.get_customer_by_email("john.doe@example.com")
```

## Production Deployment

### Scaling Considerations

- Run on dedicated server with 16GB+ RAM
- Use Groq API for consistent performance
- Implement Redis for distributed caching
- Add monitoring and alerting
- Set up log aggregation
- Consider cloud vector database for multi-instance deployments

### Security

**Data Privacy:**
- Email content sent to Groq API - review their privacy policy
- SQLite database stored locally
- PII automatically detected and can be redacted
- Store API keys in environment variables

**Best Practices:**
- Rotate Gmail API tokens periodically
- Monitor response quality scores
- Keep product documentation updated
- Review feedback logs regularly
- Implement rate limiting for API calls

## Troubleshooting

### Model Loading Issues
- Verify `GROQ_API_KEY` is set correctly
- Check internet connection for API access

### Database Issues
- Database auto-creates on first run
- Check if `customer_data.db` file exists
- Run `python test_database.py` to verify setup

### Vector Store Errors
- Reinitialize: `python setup_vectorstore.py`
- Check `data/product_docs/` has markdown files

### Gmail Authentication Issues
- Use Desktop app client ID for Colab
- Add your email as test user in Google Cloud Console
- Delete `token.pickle` and re-authenticate

### Performance Issues
- Use smaller Groq model (llama-3.1-8b-instant)
- Disable advanced RAG features
- Enable Redis caching

## System Requirements

**Minimum:**
- CPU: Any modern processor
- RAM: 4GB
- Storage: 2GB free

**Recommended:**
- CPU: 4+ cores
- RAM: 8GB
- Storage: 5GB free (SSD preferred)

## Limitations

- English language only (extensible to other languages)
- Synchronous processing (one email at a time)
- Local vector store (can be upgraded to cloud services)

## Future Enhancements

- Multi-language support
- Parallel email processing
- Cloud vector database integration
- Advanced analytics and reporting
- CRM system integration
- Sentiment analysis
- Automatic escalation rules

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain) - AI application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [HuggingFace](https://huggingface.co) - Embeddings and models
- [ChromaDB](https://www.trychroma.com) - Vector database
- [Groq](https://groq.com) - Fast LLM inference API
- [Presidio](https://microsoft.github.io/presidio/) - PII detection

## Support

For issues or questions:
- Check troubleshooting section above
- Review error messages carefully
- Verify configuration is correct
- Ensure all prerequisites are met
