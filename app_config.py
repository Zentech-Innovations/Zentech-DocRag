# app_config.py

# --- Model Definitions ---
AVAILABLE_MODELS = {
    "Google Gemini 2.5 Pro": "gemini-2.5-pro",
    "OpenAI GPT-4.1": "gpt-4.1",
    "Anthropic Claude Sonnet 4": "claude-sonnet-4-20250514",
}

DEFAULT_MODEL_NAME = "Google Gemini 2.5 Pro"


UNIFIED_SYSTEM_INSTRUCTION = (
    "You are an expert assistant. Your primary function is to provide concise and "
    "informative answers based ONLY on the context provided.\n"
    "Do not use any outside knowledge.\n\n"
    "Context Information:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context, please answer the following question:\n"
    "Question: {query_str}\n\n"
    "Answer: "
)

# QUESTIONS = [
#     "Is our data/code backed up? If so, where?",
#     "How long are CCTV recordings backed up for?",
#     "What all social media sites am i allowed to visit on the office computer?",
#     "Are employees allowed to access their work pc's remotely?",
#     "What is the complete employee on-boarding process?",
#     "What are general guidelines for setting a passsword?",
#     "What is the escalation matrix for bugs in code",
#     "What personal devices am i allowed to carry into the office",
#     "How many leaves does an employee get every year",
#     "Who do i contact if i lose my ID card",
#     "What measures do we take to prevent from cyber attacks",
# ]

QUESTIONS = [
    "Give me a detailed summary of this document.",
    "What are the best practices for communication between the FDA and an industry in advance, during or after an inspection?",
    "What is FDORA?",
    "What is the FDA's BIMO program?",
    "Who to Contact at FDA for More Information?",
]