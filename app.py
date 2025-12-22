"""
Lottery FAQ SQL Agent using LangChain
A friendly, conversational agent that helps users understand lottery data
for Powerball and Mega Millions games.
"""

import os
from urllib.parse import quote_plus
import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import InMemorySaver

# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================

# Load environment variables from .env file
load_dotenv()
print("✓ Environment variables loaded from .env file")

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_USERNAME = "lottogra_vagent"
DB_PASSWORD = "za$5ZkI@)Z83"
DB_HOST = "50.6.205.181"
DB_NAME = "lottogra_lotto_data"

# Create connection pool (optional, for direct queries if needed)
pool_name = "lottery_pool"
pool_size = 5

try:
    pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name=pool_name,
        pool_size=pool_size,
        pool_reset_session=True,
        host=DB_HOST,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        database=DB_NAME,
        auth_plugin='mysql_native_password'
    )
    print("✓ MySQL connection pool created successfully")
except Exception as e:
    print(f"✗ Error creating connection pool: {e}")

# Create SQLAlchemy connection string for LangChain
encoded_password = quote_plus(DB_PASSWORD)
db_url = f"mysql+pymysql://{DB_USERNAME}:{encoded_password}@{DB_HOST}/{DB_NAME}"

# Initialize SQLDatabase wrapper
db = SQLDatabase.from_uri(db_url)
print(f"✓ Connected to database: {DB_NAME}")
print(f"✓ Available tables: {db.get_usable_table_names()}")

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

# API key is now loaded from .env file automatically
# Make sure your .env file contains: OPENAI_API_KEY=sk-...

# Initialize your preferred LLM
try:
    # Option 1: OpenAI (default)
    model = init_chat_model("gpt-4o", temperature=0)
    print(f"✓ LLM initialized: OpenAI GPT-4o")
    
    # Option 2: Anthropic Claude (uncomment to use)
    # model = init_chat_model("claude-sonnet-4-5-20250929", temperature=0)
    # print(f"✓ LLM initialized: Claude Sonnet 4.5")
    
except Exception as e:
    print(f"✗ Error initializing LLM: {e}")
    print("   Make sure your .env file contains the correct API key:")
    print("   OPENAI_API_KEY=sk-...")
    print("   or")
    print("   ANTHROPIC_API_KEY=sk-...")
    raise

# ============================================================================
# FRIENDLY FAQ SYSTEM PROMPT
# ============================================================================

LOTTERY_FAQ_PROMPT = """
You are a friendly and helpful lottery information assistant! Your job is to help users 
understand their lottery numbers and game statistics for Powerball and Mega Millions.

TONE & PERSONALITY:
- Be warm, friendly, and conversational (not technical or robotic)
- Use everyday language, avoid technical jargon
- Be encouraging and positive about lottery numbers
- Explain concepts in simple terms (e.g., "Hot numbers are drawn more often recently")

IMPORTANT FIRST STEP - CLARIFY THE GAME:
Before answering ANY question about numbers, performance, or statistics, you MUST:
1. Check if the user mentioned "Powerball" or "Mega Millions" in their question
2. If they did NOT specify which game, politely ask them: 
   "I'd be happy to help! Just to make sure I give you the right information - 
   are you asking about Powerball or Mega Millions?"
3. Only proceed with SQL queries AFTER you know which game they're asking about

GAME IDENTIFICATION:
- Powerball → use table: mass_powerball
- Mega Millions → use table: mega_millions
- If user says "powerball" or "pb" → mass_powerball table
- If user says "mega millions" or "mm" → mega_millions table

DATABASE INFORMATION:
- Tables: mass_powerball, mega_millions
- Number columns: pos1, pos2, pos3, pos4, pos5 (these are the white ball positions)
- Date columns: date or draw_date
- DO NOT USE these columns: game_number, bonus, jackpot, winners, location, field_multiplier

RULES FOR SQL QUERIES:
1. Numbers can appear in ANY position (pos1, pos2, pos3, pos4, pos5) - check ALL positions
2. Never generate INSERT, UPDATE, DELETE, or DROP statements
3. Always use proper MySQL syntax
4. Limit results appropriately (usually top 5-10 for "best numbers" queries)

COMMON QUESTIONS & HOW TO ANSWER THEM:

📊 "What's the probability of number X?"
- Check how many times the number was drawn across all positions
- Calculate percentage: (times drawn / total draws) × 100
- Example response: "Number 7 has been drawn 45 times out of 500 total draws, 
  giving it a 9% probability of appearing in any given draw!"

🔥 "How is number X performing?" or "Is number X hot or cold?"
- Calculate how often it appears
- Hot number: appears more frequently than average
- Cold number: appears less frequently than average
- Powerball: Hot if avg_wait < 12.5 draws, Cold otherwise
- Mega Millions: Hot if avg_wait < 10 draws, Cold otherwise
- Example response: "Number 23 is running HOT! 🔥 It's been drawn every 8 draws on average, 
  which is more frequent than the typical pattern."

🎯 "What are the best numbers to pick?"
- Show the most frequently drawn numbers in recent months (last 6 months is good)
- Present as a friendly list
- Example response: "Based on the last 6 months, here are the hottest numbers: 
  7 (drawn 15 times), 23 (14 times), 12 (13 times), 45 (12 times), and 34 (11 times)!"

👥 "How are my lucky numbers doing?" (multiple numbers)
- Show frequency for each number in the specified time period
- Example response: "Let me check how your numbers are doing! In the last 6 months: 
  Number 5 appeared 8 times, 12 appeared 10 times, 23 appeared 14 times..."

SQL QUERY PATTERNS (Use these as templates):

Pattern 1 - Single number probability:
```sql
SELECT 
    COUNT(*) AS times_drawn,
    ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table})), 1) AS probability_percent
FROM {table}
WHERE pos1 = X OR pos2 = X OR pos3 = X OR pos4 = X OR pos5 = X;
```

Pattern 2 - Single number performance (hot/cold):
```sql
SELECT
    COUNT(*) AS total_draws,
    SUM(CASE WHEN pos1 = X OR pos2 = X OR pos3 = X OR pos4 = X OR pos5 = X THEN 1 ELSE 0 END) AS times_drawn,
    ROUND(COUNT(*) * 1.0 / NULLIF(SUM(CASE WHEN pos1 = X OR pos2 = X OR pos3 = X OR pos4 = X OR pos5 = X THEN 1 ELSE 0 END), 0), 1) AS avg_draws_between,
    CASE 
        WHEN COUNT(*) * 1.0 / NULLIF(SUM(CASE WHEN pos1 = X OR pos2 = X OR pos3 = X OR pos4 = X OR pos5 = X THEN 1 ELSE 0 END), 0) < {threshold} 
        THEN 'Hot' 
        ELSE 'Cold' 
    END AS status
FROM {table};
```
(Use threshold: 12.5 for Powerball, 10 for Mega Millions)

Pattern 3 - Best numbers in recent period:
```sql
SELECT 
    number, 
    COUNT(*) AS frequency
FROM (
    SELECT pos1 AS number, date FROM {table}
    UNION ALL SELECT pos2, date FROM {table}
    UNION ALL SELECT pos3, date FROM {table}
    UNION ALL SELECT pos4, date FROM {table}
    UNION ALL SELECT pos5, date FROM {table}
) AS all_numbers
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
GROUP BY number
ORDER BY frequency DESC
LIMIT 10;
```

Pattern 4 - Multiple specific numbers performance:
```sql
SELECT 
    number, 
    COUNT(*) AS times_drawn
FROM (
    SELECT pos1 AS number, date FROM {table}
    UNION ALL SELECT pos2, date FROM {table}
    UNION ALL SELECT pos3, date FROM {table}
    UNION ALL SELECT pos4, date FROM {table}
    UNION ALL SELECT pos5, date FROM {table}
) AS all_numbers
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH) 
    AND number IN (a, b, c, d, e)
GROUP BY number
ORDER BY times_drawn DESC;
```

WORKFLOW:
1. Read the user's question carefully
2. If game (Powerball/Mega Millions) is NOT specified → ASK WHICH GAME
3. If game IS specified → proceed with query
4. Check available tables (if needed)
5. Generate appropriate SQL query using the patterns above
6. Execute query and get results
7. Translate technical results into friendly, encouraging language
8. Add helpful context (what hot/cold means, how to interpret probability, etc.)

Remember: You're helping people understand their lottery chances in a fun, friendly way! 
Make the data accessible and enjoyable to read.
"""

# ============================================================================
# CREATE SQL TOOLKIT AND AGENT
# ============================================================================

# Create toolkit with database tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

print("\n✓ Available SQL Tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description[:80]}...")

# Create the SQL agent with friendly FAQ prompt
agent = create_agent(
    model,
    tools,
    system_prompt=LOTTERY_FAQ_PROMPT,
    checkpointer=InMemorySaver(),  # Enable conversation memory
)

print("\n✓ Lottery FAQ Agent created successfully!")
print("=" * 80)

# ============================================================================
# AGENT EXECUTION FUNCTIONS
# ============================================================================

def ask_lottery_question(question: str, thread_id: str = "default", verbose: bool = False):
    """
    Ask the lottery FAQ agent a question in natural language.
    
    Uses stream_mode="values" which is ideal for conversational interactions:
    - Shows progress in real-time (tool calls, thinking steps)
    - Better UX for multi-step interactions (game clarification)
    - Returns complete state snapshots at each step
    
    Args:
        question: Your question about lottery numbers or games
        thread_id: Conversation ID (keeps context for follow-up questions)
        verbose: If True, shows SQL queries and tool calls as they happen
        
    Returns:
        The agent's friendly response
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"❓ YOUR QUESTION: {question}")
        print(f"{'='*80}\n")
    
    final_response = ""
    
    # Stream mode "values" returns full state at each step - perfect for chat!
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="values",
    ):
        if "messages" in step:
            msg = step["messages"][-1]
            
            # Show tool calls if verbose mode (great for debugging)
            if verbose and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'unknown')
                    print(f"🔧 Using tool: {tool_name}")
                    
                    if 'args' in tool_call:
                        args = tool_call['args']
                        # Show SQL queries being executed
                        if 'query' in args:
                            print(f"   📝 SQL Query:")
                            print(f"   {args['query']}\n")
                        # Show other tool arguments
                        elif 'table_names' in args:
                            print(f"   📋 Checking schema for: {args['table_names']}\n")
            
            # Detect final AI response (has content but no tool calls)
            if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                final_response = msg.content
                
                # In non-verbose mode, show intermediate responses too (like game clarification)
                if not verbose and final_response:
                    print(final_response)
    
    if verbose and final_response:
        print(f"{'='*80}")
        print(f"💬 FINAL ANSWER:\n")
        print(final_response)
        print(f"\n{'='*80}\n")
    
    return final_response


def ask_lottery_question_streaming(question: str, thread_id: str = "default"):
    """
    Interactive streaming version - shows each response as it arrives.
    Perfect for chat interfaces where you want to display messages progressively.
    
    This version prints ALL agent messages (including clarification questions)
    as they stream in, making it feel more like a live conversation.
    
    Args:
        question: Your question about lottery numbers or games
        thread_id: Conversation ID (keeps context for follow-up questions)
        
    Returns:
        The agent's final response
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n💬 You: {question}\n")
    print("🤖 Agent: ", end="", flush=True)
    
    final_response = ""
    
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="values",
    ):
        if "messages" in step:
            msg = step["messages"][-1]
            
            # Print AI responses as they arrive (makes it feel more conversational)
            if hasattr(msg, 'content') and msg.content and not hasattr(msg, 'tool_calls'):
                # If this is a new response, print it
                if msg.content != final_response:
                    if final_response:  # Not the first response
                        print(f"\n🤖 Agent: {msg.content}")
                    else:  # First response
                        print(msg.content)
                    final_response = msg.content
    
    print()  # New line after conversation
    return final_response


def chat_session():
    """
    Interactive chat session with the lottery FAQ agent.
    Maintains conversation context and handles follow-up questions naturally.
    
    Type 'quit', 'exit', or 'bye' to end the session.
    """
    import uuid
    
    # Create unique session ID
    session_id = str(uuid.uuid4())[:8]
    
    print("\n" + "="*80)
    print("🎰 LOTTERY FAQ CHAT SESSION")
    print("="*80)
    print("\nHi! I'm your lottery information assistant. Ask me anything about")
    print("Powerball or Mega Millions numbers, and I'll help you out! 🎲")
    print("\nType 'quit', 'exit', or 'bye' to end our chat.\n")
    print("="*80 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("💬 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n🎰 Thanks for chatting! Good luck with your numbers! 🍀\n")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Initialize response tracking
            full_response = ""
            is_first_response = True
            
            config = {"configurable": {"thread_id": session_id}}
            
            # Use stream_mode="updates" to see tool calls AND final responses
            for step in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="updates",
            ):
                # Extract the node name and data
                for node_name, node_data in step.items():
                    if node_name == "model":  # LLM response/tool calls
                        if "messages" in node_data:
                            msg = node_data["messages"][-1]
                            
                            # Show tool calls
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call.get('name', 'unknown')
                                    print(f"  🔧 Using tool: {tool_name}")
                                    if 'args' in tool_call and 'query' in tool_call['args']:
                                        print(f"     📋 SQL: {tool_call['args']['query'][:100]}...\n")
                            
                            # Show AI response
                            elif hasattr(msg, 'content') and msg.content:
                                if is_first_response:
                                    print("🤖 Agent: ", end="", flush=True)
                                    is_first_response = False
                                
                                # Only print new content
                                new_content = msg.content[len(full_response):]
                                if new_content:
                                    print(new_content, end="", flush=True)
                                    full_response = msg.content
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\n🎰 Chat interrupted. Thanks for visiting! 🍀\n")
            break
        except Exception as e:
            print(f"\n❌ Oops, something went wrong: {e}\n")
            print("Let's try that again!\n")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎰 LOTTERY FAQ AGENT - Ask me about your lottery numbers!")
    print("="*80)
    
    print("\n💡 Example questions you can ask:\n")
    
    example_questions = [
        "What's the probability of number 7?",  # Should ask which game
        "How is my number 23 performing in Powerball?",
        "What are the best numbers to pick in Mega Millions?",
        "Are my lucky numbers 5, 12, 23, 34, 45 doing well in Powerball?",
        "Is number 17 hot or cold in Mega Millions right now?",
        "What numbers should I avoid in Powerball?",
    ]
    
    for i, q in enumerate(example_questions, 1):
        print(f"   {i}. {q}")
    
    print("\n" + "="*80)
    print("🚀 USAGE OPTIONS:")
    print("="*80)
    
    print("\n1️⃣  Basic Question (shows only the answer):")
    print("   ask_lottery_question('What is the probability of number 7?')")
    
    print("\n2️⃣  Verbose Mode (shows SQL queries and tool calls):")
    print("   ask_lottery_question('How is number 23 doing in Powerball?', verbose=True)")
    
    print("\n3️⃣  Streaming Mode (progressive display, like ChatGPT):")
    print("   ask_lottery_question_streaming('What are the best numbers in Mega Millions?')")
    
    print("\n4️⃣  Interactive Chat Session (best for back-and-forth conversation):")
    print("   chat_session()")
    
    print("\n" + "="*80)
    print("💡 TIP: Use the same thread_id for related questions to maintain context!")
    print("="*80 + "\n")
    
    # Uncomment to test the interaction flow:
    # print("\n\nTESTING STREAMING CONVERSATION:")
    # print("-" * 80)
    # thread = "test_session_1"
    # ask_lottery_question_streaming("What's the probability of number 7?", thread_id=thread)
    # ask_lottery_question_streaming("Powerball", thread_id=thread)
    
    # Or start an interactive chat:
    chat_session()