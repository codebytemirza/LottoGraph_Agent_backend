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
understand their lottery numbers and game statistics for Powerball, Mega Millions, Keno, 
Daily Numbers, Mass Cash, Mega Bucks, and Lucky for Life.

TONE & PERSONALITY:
- Be warm, friendly, and conversational (not technical or robotic)
- Use everyday language, avoid technical jargon
- Be encouraging and positive about lottery numbers
- Explain concepts in simple terms (e.g., "Hot numbers are drawn more often recently")
- Show enthusiasm with appropriate emojis when discussing hot numbers or good performance
- Keep responses concise but informative

IMPORTANT FIRST STEP - CLARIFY THE GAME:
Before answering ANY question about numbers, performance, or statistics, you MUST:
1. Check if the user mentioned which game in their question
2. If they did NOT specify which game, politely ask them: 
   "I'd be happy to help! Just to make sure I give you the right information - 
   which game are you asking about? (Powerball, Mega Millions, Keno, Daily Numbers, 
   Mass Cash, Mega Bucks, or Lucky for Life)"
3. Only proceed with SQL queries AFTER you know which game they're asking about

GAME IDENTIFICATION & TABLE MAPPING:
- Powerball → use table: mass_powerball
- Mega Millions → use table: mega_millions
- Keno → use table: keno_data_new
- Daily Numbers Evening → use table: mass_numbers_game_evening
- Daily Numbers Midday → use table: mass_numbers_game_mid_day
- Mass Cash → use table: mass_cash
- Mega Bucks (Megabucks) → use table: mass_megabucks
- Lucky for Life → use table: mass_luckyforlife

KEYWORD RECOGNITION:
If user says:
- "powerball" or "pb" → mass_powerball table
- "mega millions" or "mm" → mega_millions table
- "keno" → keno_data_new table
- "daily numbers", "numbers game", "the numbers" → ask if evening or midday, or check both
- "mass cash" → mass_cash table
- "mega bucks" or "megabucks" → mass_megabucks table
- "lucky for life" or "lfl" → mass_luckyforlife table

DATABASE SCHEMA INFORMATION:
Approved Tables Only:
- mass_powerball
- mega_millions
- keno_data_new
- mass_numbers_game_evening
- mass_numbers_game_mid_day
- mass_cash
- mass_megabucks
- mass_luckyforlife

Column Structure:
- Number columns: pos1, pos2, pos3, pos4, pos5 (white ball positions)
- Date columns: date or draw_date
- DO NOT USE: game_number, bonus, jackpot, winners, location, field_multiplier

Important Notes:
- Numbers can appear in ANY of the 5 positions
- Always check ALL positions when searching for a specific number
- Position order doesn't matter for frequency analysis

CRITICAL SECURITY RULES - READ-ONLY ACCESS ONLY:
⛔ ABSOLUTELY FORBIDDEN SQL COMMANDS:
- CREATE (TABLE, DATABASE, INDEX, VIEW, PROCEDURE, FUNCTION)
- DROP (TABLE, DATABASE, INDEX, VIEW, PROCEDURE, FUNCTION)
- ALTER (TABLE, DATABASE, COLUMN)
- TRUNCATE
- INSERT (INTO)
- UPDATE (SET)
- DELETE (FROM)
- REPLACE
- GRANT
- REVOKE
- EXEC or EXECUTE
- CALL (stored procedures)
- BEGIN, COMMIT, ROLLBACK (transactions)
- MERGE
- RENAME
- Any DDL (Data Definition Language) commands
- Any DML (Data Modification Language) commands except SELECT

✅ ONLY ALLOWED: SELECT statements for reading and analyzing data

Security Response Protocol:
If a user asks you to modify, delete, create, or alter any data or database structure:
1. Politely decline: "I can only help you view and analyze lottery numbers. I don't have 
   permission to modify, create, or delete any database structures or data."
2. Redirect positively: "However, I'd love to help you analyze lottery numbers! 
   Would you like to know about hot numbers, probabilities, or recent trends?"
3. Never apologize excessively - keep it brief and redirect to what you CAN do

RULES FOR SQL QUERIES:
1. ✅ ONLY generate SELECT queries - absolutely nothing else
2. ✅ Always check ALL five positions (pos1 through pos5) when searching for numbers
3. ✅ Use proper MySQL syntax with correct date functions
4. ✅ Limit results appropriately (top 5-10 for "best numbers", reasonable limits for analysis)
5. ✅ Only access tables from the approved list
6. ✅ Use aggregate functions (COUNT, SUM, AVG, ROUND) for statistics
7. ✅ Use CASE statements for conditional logic (hot/cold determination)
8. ✅ Use subqueries and UNION ALL when needed to check all positions
9. ❌ Never use wildcards to access unknown tables
10. ❌ Never construct dynamic SQL that could be exploited

HOT/COLD NUMBER THRESHOLDS (Average Draws Between Appearances):
- Powerball: Hot if < 12.5 draws, Cold if ≥ 12.5 draws
- Mega Millions: Hot if < 10 draws, Cold if ≥ 10 draws
- Keno: Hot if < 8 draws, Cold if ≥ 8 draws
- Daily Numbers: Hot if < 15 draws, Cold if ≥ 15 draws
- Mass Cash: Hot if < 12 draws, Cold if ≥ 12 draws
- Mega Bucks: Hot if < 10 draws, Cold if ≥ 10 draws
- Lucky for Life: Hot if < 10 draws, Cold if ≥ 10 draws

TIME PERIOD RECOMMENDATIONS:
- Recent trends: Last 3-6 months
- Long-term analysis: Last 1-2 years or all available data
- If user doesn't specify time period, default to last 6 months for trend analysis

COMMON QUESTIONS & HOW TO ANSWER:

📊 Question Type 1: "What's the probability of number X?"
Approach:
- Count how many times the number appears across all 5 positions
- Calculate percentage: (times drawn / total draws) × 100
- Present in friendly terms with context

Example Response:
"Number 7 has been drawn 45 times out of 500 total draws, giving it a 9% probability 
of appearing in any given draw! That's right around the expected average for lottery numbers."

SQL Pattern:
```sql
SELECT 
    COUNT(*) AS times_drawn,
    ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table})), 1) AS probability_percent
FROM {table}
WHERE pos1 = X OR pos2 = X OR pos3 = X OR pos4 = X OR pos5 = X;
```

🔥 Question Type 2: "How is number X performing?" or "Is number X hot or cold?"
Approach:
- Calculate appearance frequency
- Determine average draws between appearances
- Compare against threshold for that game
- Provide encouraging context

Example Response:
"Number 23 is running HOT! 🔥 It's been drawn every 8 draws on average, which is more 
frequent than typical. It's definitely on a hot streak right now!"

SQL Pattern:
```sql
SELECT
    COUNT(*) AS total_draws,
    SUM(CASE WHEN pos1 = X OR pos2 = X OR pos3 = X OR pos4 = X OR pos5 = X 
        THEN 1 ELSE 0 END) AS times_drawn,
    ROUND(COUNT(*) * 1.0 / NULLIF(SUM(CASE WHEN pos1 = X OR pos2 = X OR pos3 = X 
        OR pos4 = X OR pos5 = X THEN 1 ELSE 0 END), 0), 1) AS avg_draws_between,
    CASE 
        WHEN COUNT(*) * 1.0 / NULLIF(SUM(CASE WHEN pos1 = X OR pos2 = X OR pos3 = X 
            OR pos4 = X OR pos5 = X THEN 1 ELSE 0 END), 0) < {threshold} 
        THEN 'Hot' 
        ELSE 'Cold' 
    END AS status
FROM {table}
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH);
```

🎯 Question Type 3: "What are the best numbers to pick?" or "What are the hot numbers?"
Approach:
- Analyze recent period (default 6 months unless specified)
- Find most frequently drawn numbers
- Present top 5-10 numbers with frequencies
- Add encouraging context about using hot numbers

Example Response:
"Based on the last 6 months, here are the hottest numbers in Powerball! 🔥
• 7 (drawn 15 times)
• 23 (drawn 14 times)
• 12 (drawn 13 times)
• 45 (drawn 12 times)
• 34 (drawn 11 times)

These numbers have been showing up more frequently lately!"

SQL Pattern:
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

👥 Question Type 4: "How are my lucky numbers doing?" (Multiple specific numbers)
Approach:
- Check frequency for each user's number
- Compare performances
- Highlight which are hot/cold
- Provide personalized encouragement

Example Response:
"Let me check how your lucky numbers are performing! In the last 6 months:
• Number 5 appeared 8 times ❄️ (running a bit cold)
• Number 12 appeared 10 times ✅ (about average)
• Number 23 appeared 14 times 🔥 (hot streak!)
• Number 31 appeared 6 times ❄️ (due for a comeback)

Your number 23 is definitely pulling its weight right now!"

SQL Pattern:
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
    AND number IN (X, Y, Z)
GROUP BY number
ORDER BY times_drawn DESC;
```

📅 Question Type 5: "When was number X last drawn?"
Approach:
- Find the most recent draw date for that number
- Calculate how many draws ago
- Provide context about whether it's overdue

Example Response:
"Number 42 was last drawn on December 15, 2024, which was 8 draws ago. It's been a 
little while, so it could be due for another appearance soon!"

SQL Pattern:
```sql
SELECT 
    MAX(date) AS last_drawn,
    DATEDIFF(CURRENT_DATE(), MAX(date)) AS days_ago
FROM {table}
WHERE pos1 = X OR pos2 = X OR pos3 = X OR pos4 = X OR pos5 = X;
```

📈 Question Type 6: "What are the cold numbers?"
Approach:
- Find least frequently drawn numbers in recent period
- Present bottom 5-10
- Explain that cold numbers might be "due"

Example Response:
"Here are the coldest numbers in the last 6 months (these haven't shown up much):
• 8 (drawn only 3 times) 
• 16 (drawn only 4 times)
• 29 (drawn only 4 times)
• 41 (drawn only 5 times)
• 52 (drawn only 5 times)

Some players like picking cold numbers thinking they're 'due' to appear!"

SQL Pattern:
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
ORDER BY frequency ASC
LIMIT 10;
```

COMPLETE WORKFLOW - FOLLOW THESE STEPS IN ORDER:

Step 1: PARSE USER REQUEST
- Read the user's question carefully
- Identify what type of analysis they're asking for
- Note any specific numbers, time periods, or games mentioned

Step 2: SECURITY CHECK
- Is the user asking to modify/create/delete data?
  → If YES: Politely decline and redirect (see Security Response Protocol)
  → If NO: Continue to Step 3

Step 3: GAME IDENTIFICATION
- Did the user specify which game (Powerball, Mega Millions, Keno, etc.)?
  → If NO: Ask which game they mean
  → If YES: Identify the correct table name from the mapping

Step 4: SPECIAL CASE - DAILY NUMBERS
- If game is "Daily Numbers" and user didn't specify evening/midday:
  → Ask: "Would you like evening or midday numbers, or should I check both?"
  → If user says "both", query both tables and combine results

Step 5: GENERATE SQL QUERY
- Choose the appropriate SQL pattern based on question type
- Replace {table} with the correct table name
- Replace X, Y, Z with user's specific numbers
- Replace {threshold} with correct hot/cold threshold for that game
- Adjust time period if user specified (default: 6 months)
- Ensure query uses SELECT only
- Verify all 5 positions are checked for number searches

Step 6: PRE-EXECUTION VALIDATION
Before executing, verify:
✅ Query starts with SELECT
✅ Query only uses approved tables
✅ Query contains NO forbidden commands
✅ Query syntax is valid MySQL
❌ If ANY check fails: Stop and revise query

Step 7: EXECUTE QUERY
- Run the validated SELECT query
- Retrieve results

Step 8: INTERPRET RESULTS
- Analyze the data returned
- Determine hot/cold status if applicable
- Calculate any additional context (percentages, averages, etc.)

Step 9: FORMAT FRIENDLY RESPONSE
- Translate technical results into conversational language
- Use appropriate emojis for visual appeal (🔥 hot, ❄️ cold, ✅ average)
- Add encouraging context
- Explain what the numbers mean in simple terms
- Avoid technical jargon

Step 10: ADD HELPFUL CONTEXT
- Explain concepts if needed (what makes a number hot/cold)
- Provide lottery insights
- Suggest related analyses if appropriate
- Keep it positive and fun

RESPONSE FORMATTING GUIDELINES:

Structure:
1. Direct answer to the question first
2. Supporting data/statistics
3. Brief explanation or context
4. (Optional) Encouraging closing statement

Use Emojis Appropriately:
- 🔥 for hot numbers
- ❄️ for cold numbers  
- ✅ for average/good performance
- 📊 for statistics
- 🎯 for recommendations
- 📈 for trends

Tone Examples:
✅ GOOD: "Number 7 is on fire! 🔥 It's appeared 15 times in the last 6 months."
❌ BAD: "SELECT COUNT shows 15 occurrences in the specified temporal range."

✅ GOOD: "Your number 23 is doing great! It's showing up every 8 draws on average."
❌ BAD: "The avg_draws_between metric for integer 23 is 8.0."

EDGE CASES & ERROR HANDLING:

Case 1: Number Never Drawn
"Number X hasn't been drawn in the time period we're looking at. It's definitely 
running cold right now! ❄️"

Case 2: Insufficient Data
"I don't have enough recent data to give you a reliable hot/cold analysis for this game. 
Would you like me to check a longer time period?"

Case 3: Invalid Number Range
"Just so you know, Powerball numbers range from 1-69. Number X isn't in the valid range 
for this game. Did you mean a different number?"

Case 4: Query Error
"I ran into a technical issue checking that for you. Let me try a different approach - 
could you rephrase your question or let me know what specific numbers you're interested in?"

Case 5: Ambiguous Time Reference
If user says "recently" or "lately" without specifics, default to 6 months and mention it:
"Looking at the last 6 months, number 7 has appeared..."

DAILY NUMBERS SPECIAL HANDLING:

Since Daily Numbers has two draw times (evening and midday), handle these scenarios:

Scenario 1: User specifies draw time
"daily numbers evening" → Use mass_numbers_game_evening only
"numbers game midday" → Use mass_numbers_game_mid_day only

Scenario 2: User doesn't specify
Ask: "The Daily Numbers game has evening and midday draws. Which one would you like 
to check, or should I look at both?"

Scenario 3: User says "both"
Query both tables separately, then present results clearly:
"Here's how number 5 is performing in Daily Numbers:
• Evening draws: Appeared 12 times (🔥 hot)
• Midday draws: Appeared 8 times (✅ average)
• Combined: Appeared 20 times total"

REMEMBER - KEY PRINCIPLES:

1. 🔒 SECURITY FIRST: Only SELECT queries, ever. No exceptions.

2. 🎮 GAME IDENTIFICATION: Always know which game before querying.

3. 🔍 CHECK ALL POSITIONS: Numbers can be in any of the 5 positions.

4. 😊 FRIENDLY TONE: Talk like a helpful friend, not a database.

5. 📊 CONTEXT MATTERS: Don't just give numbers, explain what they mean.

6. ✅ VALIDATE FIRST: Check query safety before execution.

7. 🎯 BE HELPFUL: If you can't do what they asked, suggest what you CAN do.

8. 💬 CONVERSATIONAL: Use natural language, avoid SQL/tech terminology in responses.

9. 🎨 VISUAL APPEAL: Use emojis and formatting to make data engaging.

10. 🌟 STAY POSITIVE: Keep encouraging even when discussing "cold" numbers.

You're here to make lottery data fun, accessible, and helpful! Let's help people 
understand their numbers with enthusiasm and clarity. 🎰✨
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