import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from mcqgenerator.utils import read_file, get_table_data
import streamlit as st

from mcqgenerator.MCQGenerator import generate_evaluate_chain
from mcqgenerator.logger import logging
from langchain.callbacks.base import BaseCallbackHandler

import re


class SaveLLMResponseHandler(BaseCallbackHandler):
    def __init__(self):
        self.last_response = None

    def on_llm_end(self, response, **kwargs):
        self.last_response = response


def extract_usage_from_response(resp):
    if resp is None:
        return None
    meta = getattr(resp, 'response_metadata', None) or (resp.get('response_metadata') if isinstance(resp, dict) else None)
    if not meta:
        return None
    return meta.get('usage_metadata') or meta.get('usage') or meta.get('token_count') or None


def format_usage_for_display(usage):
    if not usage:
        return None
    total = usage.get('total_tokens') or usage.get('total_token_count') or usage.get('total') or usage.get('tokens')
    prompt = usage.get('prompt_tokens') or usage.get('input_token_count') or usage.get('input_tokens')
    completion = usage.get('completion_tokens') or usage.get('output_token_count') or usage.get('output_tokens')
    cost = usage.get('total_cost') or usage.get('cost')
    return {
        'total_tokens': total,
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'cost': cost,
    }


def extract_json_from_text(text):
    """Try to find a JSON object inside a larger text blob and return it as a string.
    Returns None if not found."""
    if not text or not isinstance(text, str):
        return None
    # naive approach: find the first { ... } block that looks like JSON
    # This will find the outermost JSON object by matching braces roughly.
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            stack.append(ch)
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = text[start:i+1]
                    return candidate
    return None

#loading json file

with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

#creating a title for the app
st.title("MCQs Creator Application with LangChain 🦜⛓️")

#Create a form using st.form
with st.form("user_inputs"):
    #File Upload
    uploaded_file=st.file_uploader("Uplaod a PDF or txt file")

    #Input Fields
    mcq_count=st.number_input("No. of MCQs", min_value=3, max_value=50)

    #Subject
    subject=st.text_input("Insert Subject",max_chars=20)

    # Quiz Tone
    tone=st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")

    # (diagnostics UI removed) -- app will show only the MCQ table on success

    #Add Button
    button=st.form_submit_button("Create MCQs")

    # Check if the button is clicked and all fields have input

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                # preparing payload (no UI debug text)
                # Count tokens and capture provider response metadata using a small callback handler
                # only capture LLM response; do not emit StdOut logs
                save_handler = SaveLLMResponseHandler()
                payload = {
                    "text": text,
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(RESPONSE_JSON),
                }
                # invoking chain (no UI debug text)
                # invoke the LCEL/chain with callbacks
                try:
                    response = generate_evaluate_chain.invoke(payload, config={"callbacks": [save_handler]})
                except AttributeError:
                    # fallback if generate_evaluate_chain is a callable function (older style)
                    response = generate_evaluate_chain(payload)
                    # if older style, no save_handler support
                    save_handler = None
                # chain finished (no UI debug text)

            except Exception as e:
                tb = traceback.format_exc()
                # print to server console
                traceback.print_exception(type(e), e, e.__traceback__)
                # show error in the Streamlit app for easier debugging
                st.error(f"Error: {e}")
                st.text(tb)

            else:
                # no token/usage shown in UI
                if isinstance(response, dict):
                    #Extract the quiz data from the response
                    quiz = response.get("quiz", None)
                    # if quiz is a blob of text that contains JSON, try to extract it
                    if quiz is not None:
                        # if it's a wrapper object with text
                        if not isinstance(quiz, (dict, str)):
                            # try to coerce to string
                            quiz = str(quiz)

                        # when quiz is a string, it may contain the JSON inside triple backticks or other text
                        if isinstance(quiz, str):
                            # try to find a JSON substring
                            json_candidate = extract_json_from_text(quiz)
                            if json_candidate:
                                try:
                                    quiz_parsed = json.loads(json_candidate)
                                except Exception:
                                    quiz_parsed = None
                            else:
                                # maybe the chain returned raw JSON string
                                try:
                                    quiz_parsed = json.loads(quiz)
                                except Exception:
                                    quiz_parsed = None
                        else:
                            quiz_parsed = quiz

                        table_data = None
                        if quiz_parsed is not None:
                            table_data = get_table_data(quiz_parsed)

                        if table_data and isinstance(table_data, list):
                            try:
                                df = pd.DataFrame(table_data)
                                df.index = df.index + 1
                                st.table(df)
                            except Exception as e:
                                st.error(f"Failed to create DataFrame: {e}")
                        else:
                            st.error("Unable to parse quiz into table rows.")

                else:
                    # unexpected non-dict response from chain; do not display provider output in UI
                    st.error("Generation completed but returned unexpected response format.")