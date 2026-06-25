from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = os.getenv("GCP_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it", temperature=0.4)

# Fixed Meta-Prompt Template using clean Triple Quotes
meta_prompt_template = PromptTemplate.from_template(
    """User goal: {goal}

You are an expert technical prompt engineer. Your task is to convert the user's goal into a single, concise, and direct instructional prompt for an AI coding assistant.
The prompt should be highly technical, straight to the point, and written in English.
Return ONLY the generated prompt, without any conversational text or quotes."""
)
# Chain 1: Generates the optimized prompt
prompt_generator_chain = meta_prompt_template | llm | StrOutputParser()

# Chain 2: Executes the generated prompt
execution_prompt_template = PromptTemplate.from_template("{generated_prompt}")
execution_chain = execution_prompt_template | llm | StrOutputParser()

# Execution Flow
user_goal = "Install AppsFlyer's SDK in my app using their MCP"

# 1. Generate the smart prompt
dynamic_prompt = prompt_generator_chain.invoke({"goal": user_goal})
print(f"--- Generated Prompt ---\n{dynamic_prompt}\n")
# 2. Fix: Execute the generated prompt through the second chain
final_result = execution_chain.invoke({"generated_prompt": dynamic_prompt})
print(f"--- Final Execution Result ---\n{final_result}")
