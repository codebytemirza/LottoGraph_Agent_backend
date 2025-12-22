import os
import sys
import asyncio
import time
import uuid
import json
import base64
import logging
import traceback
import dotenv
import websockets
import uvicorn
from typing import AsyncGenerator, Optional
from collections import defaultdict, deque
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI

# Import your existing agent
from app import agent
from logging.handlers import RotatingFileHandler

# ============================================================================
# FIX WINDOWS UNICODE ENCODING
# ============================================================================

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Configure production-grade logging"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # File handler with rotation (UTF-8 encoding)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'lottery_agent.log'),
        maxBytes=10485760,  # 10MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

dotenv.load_dotenv()

required_env_vars = ['OPENAI_API_KEY']
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"Missing required environment variable: {var}")

openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events"""
    # Startup
    logger.info("[STARTUP] Starting background tasks...")
    cleanup_task = asyncio.create_task(chat_session_manager.cleanup_expired_sessions())
    voice_cleanup_task = asyncio.create_task(voice_session_manager.cleanup_inactive_sessions())
    yield
    # Shutdown
    logger.info("[SHUTDOWN] Cancelling background tasks...")
    cleanup_task.cancel()
    voice_cleanup_task.cancel()

app = FastAPI(
    title="Lottery FAQ Agent",
    description="Real-time voice and text chat with lottery information",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Production configuration"""
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    WORKERS = 1 if sys.platform == 'win32' else int(os.getenv('WORKERS', 4))
    
    # Request handling
    MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', 50))
    MAX_QUEUE_SIZE = int(os.getenv('MAX_QUEUE_SIZE', 500))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 60))
    
    # Rate limiting
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 60))
    MAX_REQUESTS_PER_WINDOW = int(os.getenv('MAX_REQUESTS_PER_WINDOW', 100))
    
    # Session management
    SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 3600))
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', 30000))

config = Config()

# ============================================================================
# REQUEST QUEUE MANAGER (for chat API)
# ============================================================================

class RequestQueue:
    """Manages concurrent requests with queue and rate limiting"""
    
    def __init__(self, max_concurrent: int = config.MAX_CONCURRENT_REQUESTS):
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
        if not self._check_rate_limit(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return False
        
        async with self.lock:
            if self.active_requests < self.max_concurrent:
                self.active_requests += 1
                return True
            
            if len(self.request_queue) >= config.MAX_QUEUE_SIZE:
                logger.warning(f"Queue full: {len(self.request_queue)}/{config.MAX_QUEUE_SIZE}")
                return False
            
            event = asyncio.Event()
            self.request_queue.append(event)
            logger.info(f"Request queued. Queue size: {len(self.request_queue)}")
        
        await event.wait()
        async with self.lock:
            self.active_requests += 1
        return True
    
    async def release_slot(self, success: bool = True):
        """Release the request slot"""
        async with self.lock:
            self.active_requests -= 1
            
            if self.request_queue:
                event = self.request_queue.popleft()
                event.set()
            
            if success:
                self.requests_completed += 1
            else:
                self.requests_failed += 1
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limit"""
        now = time.time()
        requests = self.rate_limits[client_ip]
        
        while requests and requests[0] < now - config.RATE_LIMIT_WINDOW:
            requests.popleft()
        
        if len(requests) >= config.MAX_REQUESTS_PER_WINDOW:
            return False
        
        requests.append(now)
        return True
    
    async def get_stats(self):
        """Get queue statistics"""
        async with self.lock:
            return {
                "active_requests": self.active_requests,
                "queued_requests": len(self.request_queue),
                "requests_completed": self.requests_completed,
                "requests_failed": self.requests_failed,
                "max_concurrent": self.max_concurrent
            }

# ============================================================================
# SESSION MANAGERS
# ============================================================================

class ChatSessionManager:
    """Manages chat conversation sessions with timeout"""
    
    def __init__(self, session_timeout: int = 3600):
        self.sessions = {}
        self.session_timeout = session_timeout
        self.lock = asyncio.Lock()
    
    async def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id:
            async with self.lock:
                if session_id in self.sessions:
                    self.sessions[session_id]['last_activity'] = datetime.now()
                    return session_id
        
        new_session_id = str(uuid.uuid4())[:12]
        async with self.lock:
            self.sessions[new_session_id] = {
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'message_count': 0
            }
        
        return new_session_id
    
    async def cleanup_expired_sessions(self):
        """Remove expired sessions (background task)"""
        while True:
            try:
                await asyncio.sleep(300)
                async with self.lock:
                    now = datetime.now()
                    expired = [
                        sid for sid, data in self.sessions.items()
                        if (now - data['last_activity']).total_seconds() > self.session_timeout
                    ]
                    for sid in expired:
                        del self.sessions[sid]
                        logger.info(f"Cleaned up expired session: {sid}")
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
    
    async def get_stats(self):
        """Get session statistics"""
        async with self.lock:
            return {
                "active_sessions": len(self.sessions),
                "total_messages": sum(s.get('message_count', 0) for s in self.sessions.values())
            }

# ============================================================================
# REAL-TIME VOICE SESSION MANAGER
# ============================================================================

class VoiceSessionManager:
    """Manages real-time voice call sessions with live streaming"""
    
    def __init__(self):
        self.sessions = {}
        self.lock = asyncio.Lock()
    
    async def create_session(self, session_id: str) -> dict:
        """Create a new voice session for live streaming"""
        async with self.lock:
            session = {
                'session_id': session_id,
                'thread_id': str(uuid.uuid4())[:8],
                'created_at': datetime.now(),
                'is_active': True,
                'conversation_history': [],
                'last_activity': datetime.now()
            }
            self.sessions[session_id] = session
            logger.info(f"[VOICE] Session created: {session_id}")
            return session
    
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get voice session"""
        async with self.lock:
            return self.sessions.get(session_id)
    
    async def update_session(self, session_id: str, **kwargs):
        """Update session data"""
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].update(kwargs)
                self.sessions[session_id]['last_activity'] = datetime.now()
    
    async def add_to_history(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        async with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]['conversation_history'].append({
                    'role': role,
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                })
    
    async def delete_session(self, session_id: str):
        """Delete voice session"""
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"[VOICE] Session deleted: {session_id}")
    
    async def cleanup_inactive_sessions(self):
        """Remove inactive sessions (background task)"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                async with self.lock:
                    now = datetime.now()
                    inactive = [
                        sid for sid, session in self.sessions.items()
                        if (now - session['last_activity']).total_seconds() > 600  # 10 min timeout
                    ]
                    for sid in inactive:
                        del self.sessions[sid]
                        logger.info(f"[VOICE] Cleaned up inactive session: {sid}")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def get_stats(self):
        """Get voice session statistics"""
        async with self.lock:
            return {
                "active_voice_sessions": len(self.sessions),
                "total_conversations": sum(
                    len(s.get('conversation_history', [])) for s in self.sessions.values()
                )
            }

# Initialize managers after all classes are defined
request_queue = RequestQueue(max_concurrent=config.MAX_CONCURRENT_REQUESTS)
chat_session_manager = ChatSessionManager(session_timeout=config.SESSION_TIMEOUT)
voice_session_manager = VoiceSessionManager()

# ============================================================================
# STREAMING RESPONSE HANDLER
# ============================================================================

async def stream_agent_response(
    question: str,
    thread_id: str,
    timeout: int = config.REQUEST_TIMEOUT
) -> AsyncGenerator[str, None]:
    """Stream agent response with proper error handling and timeout"""
    
    start_time = time.time()
    full_response = ""
    tool_calls_shown = set()
    
    try:
        config_dict = {"configurable": {"thread_id": thread_id}}
        
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config_dict,
            stream_mode="values",
        ):
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Agent response timeout after {timeout}s")
            
            if "messages" in step:
                msg = step["messages"][-1]
                
                # Handle tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_id = f"{tool_call.get('name', 'unknown')}_{id(tool_call)}"
                        if tool_id not in tool_calls_shown:
                            tool_calls_shown.add(tool_id)
                            yield json.dumps({
                                "type": "tool_call",
                                "content": f"Using: {tool_call.get('name', 'unknown')}"
                            }) + "\n"
                
                # Handle responses
                if hasattr(msg, 'content') and msg.content:
                    if msg.content != full_response:
                        full_response = msg.content
                        yield json.dumps({
                            "type": "response",
                            "content": full_response
                        }) + "\n"
        
        # Send thread_id
        yield json.dumps({
            "type": "thread_id",
            "content": thread_id
        }) + "\n"
        
        elapsed = time.time() - start_time
        yield json.dumps({
            "type": "done",
            "elapsed_ms": round(elapsed * 1000, 2)
        }) + "\n"
        
    except TimeoutError as e:
        logger.error(f"Stream timeout: {e}")
        yield json.dumps({
            "type": "error",
            "content": f"Request timeout: {str(e)}"
        }) + "\n"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield json.dumps({
            "type": "error",
            "content": f"Stream processing failed: {str(e)}"
        }) + "\n"

# ============================================================================
# TEXT CHAT STREAMING ENDPOINT (MISSING!)
# ============================================================================

@app.post("/ask/stream")
async def ask_stream(request: dict):
    """Streaming endpoint for text chat questions using custom lottery agent"""
    
    question = request.get('question', '').strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")
    
    # Acquire slot with rate limiting
    can_process = await request_queue.acquire_slot("chat")
    if not can_process:
        raise HTTPException(status_code=429, detail="Server at capacity")
    
    # Get or create chat session
    thread_id = await chat_session_manager.get_or_create_session(
        request.get('thread_id')
    )
    
    # Increment message count
    async with chat_session_manager.lock:
        if thread_id in chat_session_manager.sessions:
            chat_session_manager.sessions[thread_id]['message_count'] += 1
    
    async def generate():
        try:
            async for chunk in stream_agent_response(question, thread_id):
                yield chunk
        except Exception as e:
            logger.error(f"Generate error: {e}")
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"
        finally:
            await request_queue.release_slot(success=True)
    
    return StreamingResponse(generate(), media_type="application/x-ndjson")

# ============================================================================
# REALTIME VOICE SESSION CLASS
# ============================================================================

class RealtimeSession:
    """Manages OpenAI Realtime session with custom lottery agent"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.thread_id = str(uuid.uuid4())[:8]
        self.created_at = datetime.now()
        self.is_active = False
        self.conversation_history = []
        self.realtime_ws = None
        self.response_in_progress = False
        self.pending_transcription = None
        
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

active_sessions = {}

# ============================================================================
# CUSTOM AGENT INTEGRATION FOR REALTIME
# ============================================================================

async def get_agent_response(question: str, thread_id: str) -> str:
    """Get response from custom lottery agent"""
    try:
        config = {"configurable": {"thread_id": thread_id}}
        final_response = ""
        
        logger.info(f"[REALTIME] Asking agent: {question}")
        
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config,
            stream_mode="values",
        ):
            if "messages" in step:
                msg = step["messages"][-1]
                if hasattr(msg, 'content') and msg.content:
                    if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                        final_response = msg.content
        
        if not final_response:
            return "I'm here to help with lottery questions! Could you please ask about Powerball or Mega Millions numbers?"
        
        logger.info(f"[REALTIME] Agent response: {final_response[:100]}...")
        return final_response
        
    except Exception as e:
        logger.error(f"[REALTIME] Agent error: {e}")
        traceback.print_exc()
        return "I encountered an error. Please try asking about lottery numbers again."

async def generate_tts_audio(text: str) -> Optional[str]:
    """Generate audio from text using OpenAI TTS API"""
    try:
        logger.info(f"[REALTIME] Generating TTS audio: {text[:50]}...")
        
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="pcm"
        )
        
        audio_bytes = await response.aread()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"[REALTIME] TTS audio generated: {len(audio_base64)} bytes")
        return audio_base64
        
    except Exception as e:
        logger.error(f"[REALTIME] TTS generation error: {e}")
        traceback.print_exc()
        return None

async def process_transcription(transcript: str, session: RealtimeSession, client_ws: WebSocket):
    """Process user transcription and get agent response"""
    try:
        session.response_in_progress = True
        
        # Get response from custom agent
        agent_response = await get_agent_response(transcript, session.thread_id)
        
        # Add to conversation history
        session.add_message("assistant", agent_response)
        
        # Send agent response text to client
        await client_ws.send_json({
            "type": "transcript",
            "text": agent_response,
            "role": "assistant"
        })
        
        logger.info(f"[REALTIME] Generating TTS for response")
        
        # Generate audio using OpenAI TTS API
        audio_base64 = await generate_tts_audio(agent_response)
        
        if audio_base64:
            # Send audio in chunks
            chunk_size = 4096
            for i in range(0, len(audio_base64), chunk_size):
                chunk = audio_base64[i:i + chunk_size]
                await client_ws.send_json({
                    "type": "audio",
                    "data": chunk
                })
                await asyncio.sleep(0.01)
            
            logger.info(f"[REALTIME] Audio sent in {(len(audio_base64) // chunk_size) + 1} chunks")
        
        # Signal response complete
        await client_ws.send_json({
            "type": "response_complete"
        })
        
    except Exception as e:
        logger.error(f"[REALTIME] Transcription processing error: {e}")
        traceback.print_exc()
    finally:
        session.response_in_progress = False

# ============================================================================
# OPENAI REALTIME API HANDLER (FIXED GRACEFUL SHUTDOWN)
# ============================================================================

async def handle_realtime_session(client_ws: WebSocket, session: RealtimeSession):
    """
    Manages OpenAI Realtime API connection with proper graceful shutdown
    """
    
    api_key = openai_client.api_key
    realtime_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    realtime_ws = None
    
    # Event to signal graceful shutdown
    shutdown_event = asyncio.Event()
    
    try:
        async with websockets.connect(
            realtime_url,
            additional_headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as realtime_ws:
            session.realtime_ws = realtime_ws
            logger.info("[REALTIME] Connected to OpenAI Realtime API")
            
            # Configure session with semantic VAD
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "model": "gpt-4o-realtime-preview-2024-12-17",
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.7,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 800
                    },
                    "temperature": 0.8,
                    "instructions": "You are a helpful lottery information assistant. Answer questions about lottery games and numbers."
                }
            }
            
            await realtime_ws.send(json.dumps(session_config))
            logger.info("[REALTIME] Session configured with server VAD")
            
            # Send session started to client
            await client_ws.send_json({
                "type": "session_started",
                "session_id": session.session_id
            })
            
            # Bidirectional communication with proper shutdown
            async def client_to_realtime():
                """Forward client audio to Realtime API"""
                try:
                    while not shutdown_event.is_set():
                        try:
                            # Use wait_for with shorter timeout for responsiveness
                            data = await asyncio.wait_for(
                                client_ws.receive_json(),
                                timeout=1.0  # 1 second timeout for responsiveness
                            )
                            
                            if data.get("type") == "audio":
                                audio_append = {
                                    "type": "input_audio_buffer.append",
                                    "audio": data["data"]
                                }
                                await realtime_ws.send(json.dumps(audio_append))
                            
                            elif data.get("type") == "end_call":
                                logger.info("[REALTIME] ✋ Call ended by user")
                                shutdown_event.set()
                                break
                                
                        except asyncio.TimeoutError:
                            # Timeout is normal - just check shutdown flag
                            continue
                        except Exception as e:
                            if "code" not in str(e):  # Ignore expected close codes
                                logger.debug(f"[REALTIME] Client receive error: {type(e).__name__}")
                            break
                            
                except Exception as e:
                    logger.debug(f"[REALTIME] Client task error: {type(e).__name__}")
                finally:
                    logger.info("[REALTIME] Client->Realtime task ended")
            
            async def realtime_to_client():
                """Process Realtime API events with custom agent"""
                try:
                    while not shutdown_event.is_set():
                        try:
                            # Use wait_for with timeout
                            message = await asyncio.wait_for(
                                realtime_ws.recv(),
                                timeout=30.0  # 30 second timeout for Realtime
                            )
                            event = json.loads(message)
                            event_type = event.get("type")
                            
                            if event_type == "session.created":
                                logger.info("[REALTIME] ✅ OpenAI session created")
                                
                            elif event_type == "input_audio_buffer.speech_started":
                                logger.info("[REALTIME] 🎤 User started speaking")
                                await client_ws.send_json({
                                    "type": "user_speaking_start"
                                })
                                
                            elif event_type == "input_audio_buffer.speech_stopped":
                                logger.info("[REALTIME] 🎤 User stopped speaking")
                                await client_ws.send_json({
                                    "type": "user_speaking_end"
                                })
                                
                            elif event_type == "conversation.item.input_audio_transcription.completed":
                                # User's speech transcribed by Whisper
                                transcript = event.get("transcript", "").strip()
                                logger.info(f"[REALTIME] 👤 User: {transcript[:60]}...")
                                
                                session.add_message("user", transcript)
                                
                                # Send user transcript to client
                                try:
                                    await client_ws.send_json({
                                        "type": "transcript",
                                        "text": transcript,
                                        "role": "user"
                                    })
                                except Exception as e:
                                    logger.debug(f"[REALTIME] Failed to send transcript: {type(e).__name__}")
                                    break
                                
                                # Process with custom agent
                                if session.response_in_progress:
                                    logger.info("[REALTIME] ⏳ Response in progress, queueing")
                                    session.pending_transcription = transcript
                                else:
                                    await process_transcription(transcript, session, client_ws)
                            
                            elif event_type == "response.done":
                                logger.info("[REALTIME] ✅ Response cycle complete")
                                session.response_in_progress = False
                                
                                # Process pending transcription if queued
                                if session.pending_transcription:
                                    transcript = session.pending_transcription
                                    session.pending_transcription = None
                                    logger.info(f"[REALTIME] 📋 Processing queued: {transcript[:60]}...")
                                    await process_transcription(transcript, session, client_ws)
                            
                            elif event_type == "error":
                                error_msg = event.get("error", {})
                                logger.error(f"[REALTIME] ❌ OpenAI error: {error_msg}")
                                try:
                                    await client_ws.send_json({
                                        "type": "error",
                                        "error": str(error_msg)
                                    })
                                except Exception as e:
                                    logger.debug(f"[REALTIME] Failed to send error: {type(e).__name__}")
                                break
                                
                        except asyncio.TimeoutError:
                            logger.debug("[REALTIME] Realtime receive timeout")
                            continue
                        except Exception as e:
                            if "code" not in str(e):  # Ignore expected close codes
                                logger.debug(f"[REALTIME] Realtime receive error: {type(e).__name__}")
                            break
                            
                except Exception as e:
                    logger.debug(f"[REALTIME] Realtime task error: {type(e).__name__}")
                finally:
                    logger.info("[REALTIME] Realtime->Client task ended")
            
            # Run both tasks concurrently with proper cleanup
            try:
                # Wait for either task to complete or shutdown event
                client_task = asyncio.create_task(client_to_realtime())
                realtime_task = asyncio.create_task(realtime_to_client())
                
                # Wait for first task to fail OR shutdown to be signaled
                done, pending = await asyncio.wait(
                    [client_task, realtime_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Signal shutdown
                shutdown_event.set()
                logger.info("[REALTIME] 🛑 Initiating graceful shutdown...")
                
                # Wait for both tasks to complete (max 5 seconds)
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("[REALTIME] ⚠️ Shutdown timeout, cancelling tasks")
                    for task in pending:
                        task.cancel()
                
            except Exception as e:
                logger.error(f"[REALTIME] ❌ Task coordination error: {type(e).__name__}")
                shutdown_event.set()
            
    except Exception as e:
        logger.error(f"[REALTIME] ❌ Session error: {type(e).__name__}: {str(e)[:100]}")
        
        try:
            await client_ws.send_json({
                "type": "error",
                "error": f"Connection failed: {type(e).__name__}"
            })
        except:
            pass
    finally:
        # Cleanup realtime connection gracefully
        if realtime_ws:
            try:
                await realtime_ws.close()
                logger.info("[REALTIME] 🔌 Realtime connection closed")
            except Exception as e:
                logger.debug(f"[REALTIME] Close error: {type(e).__name__}")
        
        logger.info("[REALTIME] ✨ Session cleaned up")

# ============================================================================
# WEBSOCKET ENDPOINT FOR REALTIME VOICE
# ============================================================================

@app.websocket("/ws/call")
async def voice_call_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice calls"""
    
    await websocket.accept()
    
    session_id = str(uuid.uuid4())[:12]
    session = RealtimeSession(session_id)
    active_sessions[session_id] = session
    
    logger.info(f"[REALTIME] 📱 Call session created: {session_id}")
    
    try:
        # Wait for start_call message
        data = await websocket.receive_json()
        
        if data.get("type") == "start_call":
            session.is_active = True
            logger.info(f"[REALTIME] 📞 Starting session: {session_id}")
            
            # Start OpenAI Realtime session
            await handle_realtime_session(websocket, session)
        
    except WebSocketDisconnect:
        logger.info(f"[REALTIME] 🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"[REALTIME] ❌ WebSocket error: {type(e).__name__}")
        
        try:
            await websocket.send_json({
                "type": "error",
                "error": f"Session error: {type(e).__name__}"
            })
        except:
            pass
    finally:
        # Cleanup session
        if session_id in active_sessions:
            del active_sessions[session_id]
            logger.info(f"[REALTIME] ✅ Session removed: {session_id}")
        
        try:
            await websocket.close()
        except:
            pass

# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_stats = await request_queue.get_stats()
    chat_stats = await chat_session_manager.get_stats()
    voice_stats = await voice_session_manager.get_stats()
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "chat": chat_stats,
        "voice": voice_stats,
        "queue": queue_stats
    }

@app.get("/stats")
async def stats():
    """Get detailed API statistics"""
    queue_stats = await request_queue.get_stats()
    chat_stats = await chat_session_manager.get_stats()
    voice_stats = await voice_session_manager.get_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "chat_management": chat_stats,
        "voice_management": voice_stats,
        "queue_management": queue_stats,
        "limits": {
            "max_concurrent": config.MAX_CONCURRENT_REQUESTS,
            "max_queue_size": config.MAX_QUEUE_SIZE,
            "request_timeout": config.REQUEST_TIMEOUT
        }
    }

@app.get("/sessions")
async def list_sessions():
    """List all active voice sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "thread_id": session.thread_id,
                "is_active": session.is_active,
                "created_at": session.created_at.isoformat(),
                "conversation_length": len(session.conversation_history)
            }
            for sid, session in active_sessions.items()
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Lottery FAQ Agent",
        "version": "1.0.0",
        "endpoints": {
            "text_chat": "POST /ask/stream",
            "voice_call": "WebSocket /ws/call",
            "health": "GET /health",
            "stats": "GET /stats",
            "sessions": "GET /sessions"
        }
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("")
    logger.info("[STARTUP] ================================================================================")
    logger.info("[STARTUP] Unified Lottery FAQ Agent - OpenAI Realtime + Custom Agent")
    logger.info("[STARTUP] ================================================================================")
    logger.info(f"[STARTUP] Server: {config.HOST}:{config.PORT}")
    logger.info("[STARTUP] Endpoints:")
    logger.info("[STARTUP]   • Text Chat:    POST http://localhost:8000/ask/stream")
    logger.info("[STARTUP]   • Voice Call:   WebSocket ws://localhost:8000/ws/call")
    logger.info("[STARTUP]   • Sessions:     GET http://localhost:8000/sessions")
    logger.info("[STARTUP]   • Health:       GET http://localhost:8000/health")
    logger.info("[STARTUP]   • Stats:        GET http://localhost:8000/stats")
    logger.info("[STARTUP] ================================================================================")
    logger.info("")
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level="debug" if config.DEBUG else "info",
        access_log=True,
        reload=config.DEBUG
    )