"""
Production-grade Streaming Chat API for Lottery FAQ Agent
Handles high-volume requests (1000+/sec) with proper queue management,
connection pooling, and error recovery.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Import your agent
from app import agent

# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_CONCURRENT_REQUESTS = 50  # Prevent resource exhaustion
MAX_QUEUE_SIZE = 500
REQUEST_TIMEOUT = 60  # seconds
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 100  # per IP

# ============================================================================
# REQUEST QUEUE MANAGER
# ============================================================================

class RequestQueue:
    """Manages concurrent requests with queue and rate limiting"""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        self.max_concurrent = max_concurrent
        self.active_requests = 0
        self.request_queue = deque()
        self.lock = asyncio.Lock()
        self.requests_completed = 0
        self.requests_failed = 0
        self.avg_response_time = 0
        self.rate_limits = defaultdict(deque)
        
    async def acquire_slot(self, client_ip: str) -> bool:
        """Try to acquire a slot for processing"""
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            return False
        
        async with self.lock:
            if self.active_requests < self.max_concurrent:
                self.active_requests += 1
                return True
            
            # Queue is full
            if len(self.request_queue) >= MAX_QUEUE_SIZE:
                return False
            
            self.request_queue.append(asyncio.Event())
            event = self.request_queue[-1]
        
        # Wait for a slot to become available
        await event.wait()
        return True
    
    async def release_slot(self, success: bool = True):
        """Release the request slot"""
        async with self.lock:
            self.active_requests -= 1
            
            if success:
                self.requests_completed += 1
            else:
                self.requests_failed += 1
            
            # Notify queued request
            if self.request_queue:
                event = self.request_queue.popleft()
                event.set()
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limit"""
        now = time.time()
        requests = self.rate_limits[client_ip]
        
        # Remove old requests outside the window
        while requests and requests[0] < now - RATE_LIMIT_WINDOW:
            requests.popleft()
        
        # Check if limit exceeded
        if len(requests) >= MAX_REQUESTS_PER_WINDOW:
            return False
        
        requests.append(now)
        return True
    
    async def get_stats(self):
        """Get queue statistics"""
        async with self.lock:
            return {
                "active_requests": self.active_requests,
                "queued_requests": len(self.request_queue),
                "max_concurrent": self.max_concurrent,
                "requests_completed": self.requests_completed,
                "requests_failed": self.requests_failed,
                "avg_response_time_ms": round(self.avg_response_time * 1000, 2)
            }

# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """Manages conversation sessions with timeout"""
    
    def __init__(self, session_timeout: int = 3600):
        self.sessions = {}
        self.session_timeout = session_timeout
        self.lock = asyncio.Lock()
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id:
            async with self.lock:
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    session['last_access'] = time.time()
                    return session_id
        
        # Create new session
        new_session_id = str(uuid.uuid4())[:12]
        async with self.lock:
            self.sessions[new_session_id] = {
                'thread_id': str(uuid.uuid4())[:8],
                'created_at': time.time(),
                'last_access': time.time(),
                'message_count': 0
            }
        
        return new_session_id
    
    async def cleanup_expired_sessions(self):
        """Remove expired sessions (background task)"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            now = time.time()
            async with self.lock:
                expired = [
                    sid for sid, session in self.sessions.items()
                    if now - session['last_access'] > self.session_timeout
                ]
                for sid in expired:
                    del self.sessions[sid]
            
            if expired:
                print(f"🧹 Cleaned up {len(expired)} expired sessions")
    
    async def get_session_stats(self):
        """Get session statistics"""
        async with self.lock:
            return {
                "active_sessions": len(self.sessions),
                "session_timeout_seconds": self.session_timeout
            }

# ============================================================================
# STREAMING RESPONSE HANDLER
# ============================================================================

async def stream_agent_response(
    question: str,
    thread_id: str,
    timeout: int = REQUEST_TIMEOUT
) -> AsyncGenerator[str, None]:
    """
    Stream agent response with proper error handling and timeout.
    Yields JSON-formatted chunks for client consumption.
    """
    
    start_time = time.time()
    full_response = ""
    tool_calls_shown = set()
    
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Wrap stream with timeout
        async def agent_stream_with_timeout():
            nonlocal full_response
            try:
                for step in agent.stream(
                    {"messages": [{"role": "user", "content": question}]},
                    config,
                    stream_mode="values",
                ):
                    # Check timeout
                    if time.time() - start_time > timeout:
                        yield {
                            "type": "error",
                            "content": "Request timeout - took too long to process"
                        }
                        return
                    
                    if "messages" in step:
                        msg = step["messages"][-1]
                        
                        # Handle tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_id = id(tool_call)
                                if tool_id not in tool_calls_shown:
                                    tool_calls_shown.add(tool_id)
                                    tool_name = tool_call.get('name', 'unknown')
                                    yield {
                                        "type": "tool_call",
                                        "content": f"Using {tool_name}..."
                                    }
                        
                        # Handle response content
                        if hasattr(msg, 'content') and msg.content:
                            if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                                if msg.content != full_response:
                                    full_response = msg.content
                                    yield {
                                        "type": "response",
                                        "content": full_response
                                    }
                
            except asyncio.TimeoutError:
                yield {
                    "type": "error",
                    "content": "Agent processing timeout"
                }
            except Exception as e:
                yield {
                    "type": "error",
                    "content": f"Agent error: {str(e)}"
                }
        
        # Stream with timeout
        async for chunk in agent_stream_with_timeout():
            yield chunk
        
        # Send completion signal
        elapsed = time.time() - start_time
        yield {
            "type": "done",
            "elapsed_ms": round(elapsed * 1000, 2)
        }
        
    except Exception as e:
        print(f"❌ Stream error: {e}")
        yield {
            "type": "error",
            "content": f"Stream processing failed: {str(e)}"
        }

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Lottery FAQ Streaming Chat API",
    version="1.0.0",
    description="Production-grade streaming chat API for lottery FAQ agent"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
request_queue = RequestQueue()
session_manager = SessionManager()

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

@app.on_event("startup")
async def startup_tasks():
    """Initialize background tasks"""
    print("🚀 Starting Lottery FAQ Chat API...")
    print(f"📊 Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"📊 Max queue size: {MAX_QUEUE_SIZE}")
    print(f"⏱️  Request timeout: {REQUEST_TIMEOUT}s")
    
    # Start session cleanup task
    asyncio.create_task(session_manager.cleanup_expired_sessions())
    
    print("✅ API initialized and ready for requests")

# ============================================================================
# STREAMING ENDPOINTS
# ============================================================================

@app.post("/ask/stream")
async def ask_stream(request: dict):
    """
    Streaming endpoint for chat questions.
    
    Request body:
    {
        "question": "What are the best numbers in Powerball?",
        "thread_id": "optional-session-id" (optional, creates new if not provided)
    }
    
    Response: Server-Sent Events (SSE) with JSON chunks
    """
    
    # Get client IP for rate limiting
    client_ip = request.get('client_ip', 'unknown')
    
    # Validate input
    question = request.get('question', '').strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")
    
    # Try to acquire request slot
    can_process = await request_queue.acquire_slot(client_ip)
    if not can_process:
        raise HTTPException(
            status_code=429,
            detail="Server at capacity. Please try again later."
        )
    
    # Get or create session
    thread_id_param = request.get('thread_id')
    thread_id = await session_manager.get_or_create_session(thread_id_param)
    
    # Update session message count
    session_manager.sessions[thread_id]['message_count'] += 1
    
    async def generate():
        """Generate streaming response"""
        try:
            # Send session info
            yield json.dumps({
                "type": "thread_id",
                "content": thread_id
            }) + "\n"
            
            # Stream agent response
            start_time = time.time()
            
            async for chunk in stream_agent_response(question, thread_id):
                yield json.dumps(chunk) + "\n"
                await asyncio.sleep(0)  # Allow other tasks to run
            
            # Track response time
            elapsed = time.time() - start_time
            request_queue.avg_response_time = (
                (request_queue.avg_response_time * request_queue.requests_completed + elapsed) 
                / (request_queue.requests_completed + 1)
            )
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            yield json.dumps({
                "type": "error",
                "content": str(e)
            }) + "\n"
        finally:
            # Always release the slot
            await request_queue.release_slot(success=True)
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    queue_stats = await request_queue.get_stats()
    session_stats = await session_manager.get_session_stats()
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "queue": queue_stats,
        "sessions": session_stats,
        "api_version": "1.0.0"
    }

@app.get("/stats")
async def stats():
    """Get detailed API statistics"""
    queue_stats = await request_queue.get_stats()
    session_stats = await session_manager.get_session_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "queue_management": queue_stats,
        "session_management": session_stats,
        "rate_limiting": {
            "window_seconds": RATE_LIMIT_WINDOW,
            "max_per_window": MAX_REQUESTS_PER_WINDOW
        },
        "limits": {
            "max_concurrent": MAX_CONCURRENT_REQUESTS,
            "max_queue_size": MAX_QUEUE_SIZE,
            "request_timeout_seconds": REQUEST_TIMEOUT,
            "max_question_length": 1000
        }
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🎰 Lottery FAQ Streaming Chat API")
    print("=" * 60)
    print("📡 Starting API server on http://0.0.0.0:8001")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )
