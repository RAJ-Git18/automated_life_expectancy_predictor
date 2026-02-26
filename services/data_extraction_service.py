import io
import pymupdf
from utils.logger import logger
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional
from schemas.request.predict_lifespan import PredictLifespanRequest
from dotenv import load_dotenv
import json
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
client = genai.Client()


class DataExtractionService:
    def __init__(self, file):
        self.file = file

    async def extract_data(self):
        pdf_bytes = await self.file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        pdf = pymupdf.open(stream=pdf_stream, filetype="pdf")
        extracted_text = ""
        for page in pdf:
            extracted_text += page.get_text()
        return extracted_text

    def get_system_prompt(self) -> str:
        return """You are a data extraction expert. Your task is to extract information from the following text and return it in JSON format as per the schema.
Example:
{{
  "adult_mortality": 10,
  "infant_deaths": 0,
  "under_five_deaths": 0,
  "alcohol": 194,
  "percentage_expenditure": 0,
  "hepatitis_b": 553,
  "measles": 0,
  "bmi": 34,
  "polio": 19,
  "total_expenditure": 226,
  "diphtheria": 19,
  "hiv_aids": 0,
  "gdp": 448,
  "population": 652,
  "thinness_1_19_years": 34,
  "thinness_5_9_years": 34,
  "income_composition_of_resources": 167,
  "schooling": 12.4
}}
        """

    def get_user_prompt(self, extracted_text: str) -> str:
        return f"Extracted Text:\n{extracted_text}"

    async def create_request_model(self) -> PredictLifespanRequest:
        extracted_text = await self.extract_data()
        system_content = self.get_system_prompt()
        user_content = self.get_user_prompt(extracted_text)
        prompt = [
            ("system", system_content),
            ("human", user_content),
        ]
        logger.info(f"Prompt ----------> \n{prompt}")
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        structured_model = model.with_structured_output(
            schema=PredictLifespanRequest.model_json_schema(), method="json_schema"
        )
        response = structured_model.invoke(prompt)
        logger.info(f"Type of response: {type(response)}")
        logger.info(f"Response: {json.dumps(response, indent=2)}")
        return PredictLifespanRequest(**response)
