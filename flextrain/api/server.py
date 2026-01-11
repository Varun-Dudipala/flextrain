"""FastAPI server for FlexTrain dashboard."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="FlexTrain Dashboard", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FlexTrain Dashboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #333; }
            .card { background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric { font-size: 2em; font-weight: bold; color: #2196F3; }
            .label { color: #666; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>FlexTrain Dashboard</h1>
            <div class="card">
                <div class="label">Status</div>
                <div class="metric">Ready</div>
            </div>
            <div class="card">
                <div class="label">Active Jobs</div>
                <div class="metric">0</div>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/jobs")
async def list_jobs():
    """List training jobs."""
    return {"jobs": []}
