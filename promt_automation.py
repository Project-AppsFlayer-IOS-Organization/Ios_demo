import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. CHANGED: Import the OpenAI client instead of Google
from langchain_openai import ChatOpenAI

def generate_dynamic_prompt(goal: str) -> str:
    """
    Generates an optimized technical prompt based on a raw user goal.
    Automatically uses OPENAI_API_KEY from the environment.
    """
    # 2. CHANGED: Initialize the OpenAI model (using gpt-4o or gpt-4o-mini for speed/cost)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # Fixed Meta-Prompt Template using clean Triple Quotes
    meta_prompt_template = PromptTemplate.from_template(
        """User goal: {goal}

You are an expert technical prompt engineer. Your task is to convert the user's goal into a single, concise, and direct instructional prompt for an AI coding assistant.
The prompt should be highly technical, straight to the point, and written in English.
Return ONLY the generated prompt, without any conversational text or quotes."""
    )

    # Generate the optimized prompt
    prompt_generator_chain = meta_prompt_template | llm | StrOutputParser()
    return prompt_generator_chain.invoke({"goal": goal})