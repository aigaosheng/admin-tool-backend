import os
import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json
import io
import base64
import pandas as pd
from datetime import datetime
from PIL import Image
from copy import copy
from pydantic import BaseModel

# Add project paths
# rpth = Path(os.getcwd())
# sys.path.append(str(rpth))
# for v in rpth.iterdir():
#     if v.is_dir() and v.name != "__pycache__":
#         sys.path.append(str(v))

from dotenv import load_dotenv
load_dotenv()

from doc_parser.agentTaxInvoice import TaxInvoiceParser
from audit_logger.audit_logger import logger_audit_handler as logger
from functools import lru_cache

# Configuration
TEMP_DATA_PATH = os.getenv("TEMP_PATH", "")
if TEMP_DATA_PATH:
    os.makedirs(TEMP_DATA_PATH, exist_ok=True)
ENABLE_BYTE_PARSER = True

# Default invoice data structure
DEFAULT_INVOICE_DATA = {
    'date_pickup': "", 
    'id': "", 
    'passenger': "", 
    'profile': "", 
    'location_pickup': "", 
    'destination': "", 
    'fare': 0.0, 
    'platform_fee': 0.0, 
    'driver_fee': 0.0, 
    'total_paid': 0.0, 
    'currency': "", 
    'filename': "",
    'page_id': -1, 
    'status': 0,
    'image': ''
}

# Pydantic models
class InvoiceData(BaseModel):
    date_pickup: str = ""
    id: str = ""
    passenger: str = ""
    profile: str = ""
    location_pickup: str = ""
    destination: str = ""
    fare: float = 0.0
    platform_fee: float = 0.0
    driver_fee: float = 0.0
    total_paid: float = 0.0
    currency: str = ""
    filename: str = ""
    page_id: int = -1
    status: int = 0
    image: str = ''

class ProcessResponse(BaseModel):
    success: bool
    message: str
    data: List[InvoiceData]
    total_processed: int
    success_count: int
    failure_count: int

class ExportRequest(BaseModel):
    invoices: List[Dict[str, Any]]
    format: str = "excel"  # excel or json

# Initialize FastAPI app
app = FastAPI(
    title="Taxi Invoice Processing API",
    description="API for processing taxi invoice images using AI agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def init_agent_doc():
    """Initialize the document parser agent"""
    model_provider = os.getenv("LLM_PROVIDER", "ollama")
    model_name = os.getenv("MODEL_NAME", "gemma3")
    try:
        print(f"Initializing tax invoice parser with {model_provider} : {model_name}...")
        parser_ollama = TaxInvoiceParser(model_provider=model_provider, model_name=model_name)        
        return parser_ollama
    except Exception as e:
        raise Exception(f"Error initializing parser: {e}")

def documentParser(file_content: bytes, filename: str, doc_type: str):
    """Parse document and extract invoice data"""
    parser_ollama = init_agent_doc()
    try:
        metadata = parser_ollama.documentParser(file_content, filename, doc_type)
        return metadata
    except Exception as e:
        logger.error(f"Error parsing document {filename}: {e}")
        return None

def process_single_file(file_content: bytes, filename: str) -> List[Dict[str, Any]]:
    """Process a single file and extract invoice data"""
    column_num = ['fare', 'platform_fee', 'driver_fee', 'total_paid', "status"]
    
    try:
        # Determine document type
        sfx = Path(filename).suffix.strip(".").lower()
        if sfx in ("png", "jpg", "jpeg"):
            doc_type = "image"
        elif sfx in ('pdf'):
            doc_type = "pdf"
        elif sfx in ('docx'):
            doc_type = "doc"
        else:
            doc_type = "other"

        # Parse document
        parsed_data_list = documentParser(file_content, filename, doc_type)
        print(f"** parsed_data_list -> {parsed_data_list}")
        
        invoice_data = []
        if parsed_data_list:
            for parsed_data0 in parsed_data_list:
                print(f"** parsed_data0 -> {parsed_data0}")
                name = parsed_data0["name"]
                try:
                    pid = int(parsed_data0["page"].split("_")[-1])
                except:
                    pid = -1
                parsed_data = parsed_data0["metadata"]

                if parsed_data:
                    for v in parsed_data.transactions:
                        v = v.model_dump()
                        if v["total_paid"]:
                            v["status"] = 1
                        else: 
                            v["status"] = 0

                        for ky, kv in v.items():
                            if ky in column_num:
                                v[ky] = kv if kv else 0.0
                            else:
                                v[ky] = kv if kv else ""
                        
                        v["filename"] = f"{name}-page{pid}"
                        v["page_id"] = pid
                        # print(f"** v00 -> {v}")
                        try:
                            if parsed_data0.get("image"):
                                v["image"] = base64.b64encode(parsed_data0.get("image", b""))
                            else:
                                v["image"] = ""
                        except Exception as e:
                            v["image"] = ""
                            print(f"** Faile -> {parsed_data0}, {e}")
                        # print(f"** v -> {v}")

                        invoice_data.append(v)
                else:
                    invoice_data_tmp = copy(DEFAULT_INVOICE_DATA)
                    invoice_data_tmp["filename"] = name
                    invoice_data_tmp["page_id"] = pid
                    invoice_data_tmp["status"] = 0
                    invoice_data_tmp["image"] = ""
                    invoice_data.append(invoice_data_tmp)
                    logger.warning(f"Parsing failed for {name}")

            logger.info(f"Successfully processed {filename} -> {len(invoice_data)} invoices")
        else:
            invoice_data = copy(DEFAULT_INVOICE_DATA)
            invoice_data["filename"] = filename
            invoice_data["page_id"] = -1
            invoice_data["status"] = 0
            invoice_data["image"] = ""
            invoice_data = [invoice_data]
            logger.warning(f"No data extracted from {filename}")

        return invoice_data
    
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return [{**DEFAULT_INVOICE_DATA, "filename": filename, "status": 0}]

def create_excel_bytes(invoices_data: List[Dict]) -> bytes:
    """Create Excel file from invoice data"""
    if not invoices_data:
        # Create empty DataFrame with expected columns if no data
        columns = [
            'date_pickup', 'id', 'passenger', 'profile', 'location_pickup', 
            'destination', 'fare', 'platform_fee', 'driver_fee', 'total_paid', 
            'currency', 'filename', 'status'
        ]
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(invoices_data)
        
        # Remove image column for export
        if 'image' in df.columns:
            df = df.drop(columns=['image'])
        
        # Reorder columns
        column_order = [
            'date_pickup', 'id', 'passenger', 'profile', 'location_pickup', 
            'destination', 'fare', 'platform_fee', 'driver_fee', 'total_paid', 
            'currency', 'filename', 'status'
        ]
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['fare', 'platform_fee', 'driver_fee', 'total_paid']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Create Excel file in memory with proper engine
    output = io.BytesIO()
    try:
        # Use xlsxwriter engine for better compatibility
        with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name='Taxi Invoices', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Taxi Invoices']
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if cell.value:
                            max_length = max(max_length, len(str(cell.value)))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Format numeric columns
            from openpyxl.styles import NamedStyle
            currency_style = NamedStyle(name="currency")
            currency_style.number_format = '"$"#,##0.00'
            
            # Apply currency formatting to monetary columns
            if not df.empty:
                for col_idx, col_name in enumerate(df.columns, 1):
                    if col_name in ['fare', 'platform_fee', 'driver_fee', 'total_paid']:
                        for row_idx in range(2, len(df) + 2):  # Start from row 2 (after header)
                            cell = worksheet.cell(row=row_idx, column=col_idx)
                            cell.number_format = '"$"#,##0.00'
        
        # Get the bytes and reset position
        output.seek(0)
        excel_bytes = output.getvalue()
        output.close()
        
        return excel_bytes
        
    except Exception as e:
        output.close()
        logger.error(f"Error creating Excel file: {str(e)}")
        raise Exception(f"Failed to create Excel file: {str(e)}")
    finally:
        if not output.closed:
            output.close()

# API Routes

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Taxi Invoice Processing API is running"}

@app.get("/health")
async def health_check():
    """Health check with system status"""
    try:
        # Test parser initialization
        parser = init_agent_doc()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "parser_initialized": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "parser_initialized": False
        }

@app.post("/process-invoices", response_model=ProcessResponse)
async def process_invoices(files: List[UploadFile] = File(...)):
    """Process multiple invoice files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    processed_invoices = []
    success_count = 0
    failure_count = 0
    
    try:
        for file in files:
            # Read file content
            file_content = await file.read()
            
            # Process the file
            invoice_data = process_single_file(file_content, file.filename)
            processed_invoices.extend(invoice_data)

            # print(f"** invoice_data -> {invoice_data}")
            
            # Count successes and failures
            for invoice in invoice_data:
                if invoice.get('status', 0) == 1:
                    success_count += 1
                else:
                    failure_count += 1
            
            # Reset file pointer for potential reuse
            await file.seek(0)
        
        return ProcessResponse(
            success=True,
            message=f"Processed {len(files)} files successfully",
            data=[InvoiceData(**invoice) for invoice in processed_invoices],
            total_processed=len(processed_invoices),
            success_count=success_count,
            failure_count=failure_count
        )
    
    except Exception as e:
        logger.error(f"Error processing invoices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing invoices: {str(e)}")

@app.post("/export")
async def export_data(request: ExportRequest):
    """Export invoice data in specified format"""
    try:
        if request.format.lower() == "excel":
            excel_data = create_excel_bytes(request.invoices)
            filename = f"taxi_invoices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            # Create response with proper headers for Excel file
            response = Response(
                content=excel_data,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Length": str(len(excel_data)),
                    "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                }
            )
            return response
        
        elif request.format.lower() == "json":
            # Remove image data for JSON export
            clean_invoices = []
            for invoice in request.invoices:
                clean_invoice = {k: v for k, v in invoice.items() if k != 'image'}
                clean_invoices.append(clean_invoice)
            
            json_data = json.dumps(clean_invoices, indent=2)
            filename = f"taxi_invoices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            response = Response(
                content=json_data.encode('utf-8'),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Length": str(len(json_data.encode('utf-8'))),
                    "Content-Type": "application/json"
                }
            )
            return response
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
    
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

@app.get("/image/{filename}")
async def get_image(filename: str, page_id: Optional[int] = None):
    """Get image data for a specific filename and page"""
    # This would need to be implemented based on your storage strategy
    # For now, return a placeholder response
    return {"message": f"Image endpoint for {filename}, page {page_id}"}

@app.post("/validate-data")
async def validate_invoice_data(invoices: List[Dict[str, Any]]):
    """Validate invoice data"""
    errors = []
    warnings = []
    
    try:
        df = pd.DataFrame(invoices)
        
        # Check for negative values in monetary columns
        monetary_cols = ['fare', 'platform_fee', 'driver_fee', 'total_paid']
        for col in monetary_cols:
            if col in df.columns:
                negative_values = df[df[col] < 0]
                if not negative_values.empty:
                    errors.append(f"Negative values found in {col} column")
        
        # Check for empty passenger names
        if 'passenger' in df.columns:
            empty_passengers = df[df['passenger'].isna() | (df['passenger'] == '')]
            if not empty_passengers.empty:
                warnings.append(f"Empty passenger names found")
        
        # Check total_paid consistency
        if all(col in df.columns for col in ['fare', 'platform_fee', 'driver_fee', 'total_paid']):
            calculated_total = df['fare'] + df['platform_fee'] + df['driver_fee']
            inconsistent = df[abs(df['total_paid'] - calculated_total) > 0.01]
            if not inconsistent.empty:
                warnings.append(f"Total paid doesn't match sum of components")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")

SERVERLESS_DEPLOY_AWS = True
if SERVERLESS_DEPLOY_AWS:
    from mangum import Mangum

    # existing app = FastAPI(...) stays the same

    handler = Mangum(app)    
else:
    if __name__ == "__main__":
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )