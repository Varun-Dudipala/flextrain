"""FastAPI server for FlexTrain dashboard."""

import os
import json
from pathlib import Path
from typing import Dict, List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="FlexTrain Dashboard", version="0.1.0")

# Simple file-based job tracking
JOBS_FILE = Path.home() / ".flextrain" / "jobs.json"


def _load_jobs() -> List[Dict]:
    """Load jobs from file."""
    if not JOBS_FILE.exists():
        return []
    try:
        with open(JOBS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_jobs(jobs: List[Dict]) -> None:
    """Save jobs to file."""
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(JOBS_FILE, "w") as f:
        json.dump(jobs, f)


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
    jobs = _load_jobs()
    # Filter out completed jobs older than 24h
    active_jobs = [j for j in jobs if j.get("status") != "completed"]
    return {"jobs": active_jobs, "total": len(jobs)}
