import os
import PyPDF2
import json
import traceback
from io import BytesIO

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            # Streamlit's uploaded file is a file-like object; read bytes and use PdfReader
            data = file.read()
            pdf_reader = PyPDF2.PdfReader(BytesIO(data))
            text = ""
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except Exception:
                    # skip page if text extraction fails for that page
                    continue
            return text

        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            raise Exception("error reading the PDF file: " + str(e))
        
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    
    else:
        raise Exception(
            "unsupported file format only pdf and text file suppoted"
            )

def get_table_data(quiz_str):
    try:
        # Accept either a dict or a JSON string
        if isinstance(quiz_str, dict):
            quiz_dict = quiz_str
        else:
            quiz_dict = json.loads(quiz_str)

        quiz_table_data = []

        # iterate over the quiz dictionary and extract the required information
        for key, value in quiz_dict.items():
            # guard against missing keys and different shapes
            mcq = (value.get("mcq") or value.get("question") or "").strip()

            # skip entries without a usable question
            if not mcq:
                continue

            options_obj = value.get("options", {})
            # support options as dict (a: text) or list [opt1, opt2, ...]
            if isinstance(options_obj, dict):
                options = " || ".join([f"{opt}-> {text}" for opt, text in options_obj.items()])
            elif isinstance(options_obj, list):
                options = " || ".join([f"{idx+1}-> {text}" for idx, text in enumerate(options_obj)])
            else:
                options = str(options_obj)

            correct = value.get("correct") or value.get("answer") or ""
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        return quiz_table_data

    except Exception as e:
        # log the traceback for debugging and return None so callers won't try to build a DataFrame
        traceback.print_exception(type(e), e, e.__traceback__)
        return None