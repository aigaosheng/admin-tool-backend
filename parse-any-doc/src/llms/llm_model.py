from dotenv import load_dotenv
load_dotenv()
import os
import sys
from pathlib import Path

import json
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from datetime import datetime
from PIL import Image

# Pydantic imports
from pydantic import BaseModel, Field, validator
from typing import List, Optional

# LangChain imports
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI

# langchain imports
from langchain.output_parsers import PydanticOutputParser

from audit_logger import logger_audit_handler as logger
from langchain_core.output_parsers.string import StrOutputParser
from util_tool.util import *

def prepare_prompt_template(system_prompt: str = "", output_format_prompt = ""):
    """
    Args:
        system_prompt (str, optional): Additional system instructions to prepend to the prompt. Defaults to "".
    Returns:: parameterize of prompt template 
    """

    if output_format_prompt:
        output_format_prompt = f"{output_format_prompt}\n\nGenerate reply as JSON: "

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""{system_prompt}

{context}

{format_instructions}
""",
    partial_variables={"system_prompt": system_prompt, "format_instructions": output_format_prompt}
    )

    return prompt_template

class LlmModel:
    def __init__(self, provider, model_name, system_prompt: str = "", response_data_model = None):
        if provider == "ollama":
            self.llm = ChatOllama(model=model_name, temperature = 0.1, base_url = os.getenv("OLLAMA_URL") )
        elif provider == "gemini":
            self.llm = GoogleGenerativeAI(model=model_name, temperature = 0.1, api_key=os.getenv("GOOGLE_API_KEY"))
        elif provider == "openai":
            self.llm = OpenAI(model=model_name, temperature = 0.1, api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider

        self.parser = None
        if response_data_model:
            self.parser = PydanticOutputParser(pydantic_object=response_data_model)
            output_format_prompt = self.parser.get_format_instructions()
        else:
            self.parser = StrOutputParser()
            output_format_prompt = ""

        self.prompt = prepare_prompt_template(system_prompt=system_prompt, output_format_prompt = output_format_prompt)
        
        self.chain = self.llm | self.parser

    async def __call__(self, query, image: str = ""):
        if image:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self.prompt.format(context=query),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ]
            )
        else:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self.prompt.format(context=query),
                    },
                ])
        try:
            rsp = await self.chain.ainvoke([message])
            logger.info(f"Success: {query} -> {rsp}")
        except Exception as e:
            rsp = ""
            logger.warning(f"Fail: {query} -> {e}")

        return rsp

    def invoke(self, query, image: str = ""):
        if image:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self.prompt.format(context=query),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ]
            )
        else:
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": self.prompt.format(context=query),
                    },
                ])
        try:
            rsp = self.chain.invoke([message])
            logger.info(f"Success: {query} -> {rsp}")
        except Exception as e:
            rsp = ""
            logger.warning(f"Fail: {query} -> {e}")

        return rsp

# === EXAMPLE USAGE ===
async def main():
    system_prompt = "Evaluate the following action query and its output for potential risks and safety concerns."
    # system_prompt = "describe the image"

    provider = "gemini" #"ollama"
    model_name="gemini-1.5-flash" #"gemma3"

    # provider = "ollama"
    # model_name="gemma3"

    container_agent = LlmModel(provider=provider, model_name=model_name, system_prompt=system_prompt, response_data_model = "")

    img = Image.open("/home/gs/Downloads/image2.jpg")
    b64_img = pil_image2base64(img, base64_only=True)

    test_cases = [
        {
            "query":"",
            "output": "Shut down power station in Queenstown and diverse to changqi in SP Group",
            # "image": b64_img
        },
    ]

    print("Testing")
    print("=" * 50)
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        rpt = await container_agent(
            query=test_case["query"] + "\n" + test_case["output"],
            image = test_case.get("image", None),
        )
        print(rpt)
        break


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    # main()