"""
FastAPI Web Application - Table Data Visualization Q&A AI Agent
"""
PORT = 8002

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
import json
import os
import webbrowser
import threading
from typing import Optional, List
from loguru import logger

from table_agent import TableAgent
from config import settings

# Create FastAPI application
app = FastAPI(
    title="Table Data Visualization Q&A AI Agent",
    description="Intelligent table data analysis and visualization Q&A system",
    version="1.0.0"
)

# Create template and static file directories
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global AI Agent instance
table_agent = None

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    global table_agent
    table_agent = TableAgent()
    logger.info("AI Agent startup completed")
    # Auto-open browser after a short delay
    threading.Thread(target=lambda: (__import__('time').sleep(1.5), webbrowser.open(f"http://localhost:{PORT}")), daemon=True).start()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    from fastapi.responses import Response
    response = templates.TemplateResponse(request, "index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload table file"""
    try:
        # Check file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls', '.json')):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Save file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load data
        result = table_agent.load_table(file_path)
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "message": "File uploaded successfully",
                "data_summary": result["data_summary"],
                "file_path": file_path
            })
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def process_query(
    query: str = Form(...)
):
    """Process user query (auto-routes between GraphRAG and QA)"""
    try:
        if table_agent.current_data is None:
            raise HTTPException(status_code=400, detail="Please upload a table file first")

        result = table_agent.process_query(query)
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        import traceback
        logger.error(f"Query processing failed: {e}")
        logger.error(f"Full error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data-preview")
async def get_data_preview(max_rows: int = 10):
    """Get data preview"""
    try:
        if table_agent.current_data is None:
            raise HTTPException(status_code=400, detail="Please upload a table file first")
        
        preview = table_agent.get_data_preview(max_rows)
        return JSONResponse(content=preview)
        
    except Exception as e:
        logger.error(f"Failed to get data preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/column-info")
async def get_column_info():
    """Get column info"""
    try:
        if table_agent.current_data is None:
            raise HTTPException(status_code=400, detail="Please upload a table file first")
        
        column_info = table_agent.get_column_info()
        return JSONResponse(content=column_info)
        
    except Exception as e:
        logger.error(f"Failed to get column info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualize")
async def create_visualization(
    chart_type: str = Form(...),
    x_column: Optional[str] = Form(None),
    y_column: Optional[str] = Form(None),
    title: Optional[str] = Form(None)
):
    """Create visualization chart"""
    try:
        if table_agent.current_data is None:
            raise HTTPException(status_code=400, detail="Please upload a table file first")
        
        # Create visualization based on chart type
        if chart_type == "histogram" and y_column:
            result = table_agent.visualization_engine.create_histogram(
                table_agent.current_data, y_column, title=title
            )
        elif chart_type == "scatter" and x_column and y_column:
            result = table_agent.visualization_engine.create_scatter_plot(
                table_agent.current_data, x_column, y_column, title=title
            )
        elif chart_type == "bar" and x_column and y_column:
            result = table_agent.visualization_engine.create_bar_chart(
                table_agent.current_data, x_column, y_column, title=title
            )
        elif chart_type == "pie" and x_column:
            result = table_agent.visualization_engine.create_pie_chart(
                table_agent.current_data, x_column, title=title
            )
        elif chart_type == "heatmap":
            result = table_agent.visualization_engine.create_heatmap(
                table_agent.current_data, title=title
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported chart type or missing required parameters")
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Failed to create visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suggestions")
async def get_visualization_suggestions():
    """Get visualization suggestions"""
    try:
        if table_agent.current_data is None:
            raise HTTPException(status_code=400, detail="Please upload a table file first")
        
        suggestions = table_agent.visualization_engine.get_visualization_suggestions(
            table_agent.current_data
        )
        return JSONResponse(content=suggestions)
        
    except Exception as e:
        logger.error(f"Failed to get visualization suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/export-report")
async def export_analysis_report(
    query: str = Form(...),
    output_format: str = Form("json")
):
    """Export analysis report"""
    try:
        if table_agent.current_data is None:
            raise HTTPException(status_code=400, detail="Please upload a table file first")
        
        output_path = f"reports/analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        os.makedirs("reports", exist_ok=True)
        
        result = table_agent.export_analysis_report(query, output_path)
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "message": "Report exported successfully",
                "output_path": output_path
            })
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Failed to export report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    try:
        status = table_agent.get_system_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-synonym")
async def add_synonym_mapping(
    key: str = Form(...),
    synonyms: str = Form(...)
):
    """Add synonym mapping"""
    try:
        synonyms_list = [s.strip() for s in synonyms.split(',')]
        table_agent.add_custom_synonym_mapping(key, synonyms_list)
        
        return JSONResponse(content={
            "success": True,
            "message": "Synonym mapping added successfully"
        })
        
    except Exception as e:
        logger.error(f"Failed to add synonym mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)



