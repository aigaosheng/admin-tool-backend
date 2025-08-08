# Parse Any Doc

A Python package for parsing various document formats including PDF, Word, and Excel files.

## Installation

```bash
pip install parse-any-doc
```

## Usage

```
# Set .env

MODEL_NAME = "gemma3"
LLM_PROVIDER = "ollama"
# MODEL_NAME = "gemini-1.5-flash"
# LLM_PROVIDER = "gemini"
APP_SECRET_KEY = "abc123"

OLLAMA_URL = "http://localhost:11434" 

```

```python
>>> import os
>>> from dotenv import load_dotenv
>>> load_dotenv()
>>> import audit_logger
>>> from doc_parser.agentTaxInvoice import TaxInvoiceParser
>>> import llms

>>> def init_agent_doc():
>>>     """Initialize the document parser agent"""
>>>     model_provider = os.getenv("LLM_PROVIDER", "ollama")
>>>     model_name = os.getenv("MODEL_NAME", "gemma3")
>>>     try:
>>>         print(f"Initializing tax invoice parser with {model_provider} : {model_name}...")
>>>         parser_ollama = TaxInvoiceParser(model_provider=model_provider, model_name=model_name)        
>>>         return parser_ollama
>>>     except Exception as e:
>>>         raise Exception(f"Error initializing parser: {e}")

>>> def documentParser(file_content: bytes, filename: str, doc_type: str):
>>>     """Parse document and extract invoice data"""
>>>     parser_ollama = init_agent_doc()
>>>     try:
>>>         metadata = parser_ollama.documentParser(file_content, filename, doc_type)
>>>         return metadata
>>>     except Exception as e:
>>>         print(f"Error parsing document {filename}: {e}")
>>>         return None

>>> resume_file = "/home/gs/Downloads/reconcile_data/0525/New Microsoft Word Document.docx"  # Replace with your resume file path
>>> resume_file = "/home/gs/Downloads/reconcile_data/0525/New Microsoft Word Document1.pdf"
>>> resume_file = "/home/gs/Downloads/reconcile_data/0525/New Microsoft Word Document.docx"
>>> resume_file = "/home/gs/Downloads/reconcile_data/0525/照片 28-05-25 16 39 22.png"

>>> with open(resume_file, "rb") as fo:
>>>     imgbyte = fo.read()
>>> doc_type = "image" #"doc", "pdf", "image"
>>> rsp = documentParser(imgbyte, resume_file, doc_type)

>>> rsp

>>> [{'name': '/home/gs/Downloads/reconcile_data/0525/照片 28-05-25 16 39 22.png',
>>>   'page': 'page_1',
>>>   'text': 'Picked Iupon28April2025\nBooking ID: A-7Q9VDWJWWQ9RAV\n0\nTotal Paid SGD8.90\n冀 5.0\n1\nCompliments for driver\nClean & Comfy\nBreakdown\nFare 7.50\nPlatform & partner fee 0.90\nDriver Fee 0.50\nDaid\n日\n删除 分享 回复转发 更多\n',
>>>   'image': None,
>>>   'metadata': TaxInvoiceModel(transactions=[TransactionModel(date_pickup='Picked Iupon28April2025', id='A-7Q9VDWJWWQ9RAV', passenger=None, profile=None, location_pickup=None,destination=None, fare=7.5, platform_fee=0.9, driver_fee=0.5, total_paid=8.9, currency='SGD')])}]

```

## Supported Formats
- PDF
- Word Documents (.docx)
- Excel Files (.xlsx)

## To build and install the package locally
Run these commands in the terminal

```

python -m pip install --upgrade pip
python -m pip install --upgrade build
python -m build
pip install -e .

```