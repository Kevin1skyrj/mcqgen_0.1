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

# small CSS to make MCQ cards look nicer
st.markdown(
    """
    <style>
    .mcq-card { background: linear-gradient(180deg,#ffffff,#f7fbff); padding:16px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.08); margin-bottom:12px; color: #0b2540 !important; }
    .mcq-qnum { font-weight:700; color:#0b57d0; }
    .mcq-question { font-size:16px; margin-top:6px; margin-bottom:8px; color: #0b2540 !important; }
    .mcq-choice { padding:10px 10px; border-radius:8px; margin:6px 0; background:#ffffff; border:1px solid #dbeefb; color:#0b2540 !important }
    .mcq-choice:hover { background:#eef8ff; }
    .mcq-choice.selected { background:#e6f7ff; border-color:#8ed0ff }
    .mcq-correct { color: #076f07; font-weight:600 }
    .mcq-wrong { color: #a10b0b; font-weight:600 }
    .mcq-card details summary { cursor: pointer; color: #0b2540 !important; font-weight:600 }
    .mcq-card details div { color: #0b2540 !important }
    /* make the main block use full width */
    .block-container{max-width:100% !important; padding:1rem 2rem !important}
    .mcq-section { width: 100%; }
    /* override Streamlit h1 padding (increase top padding to 3rem) */
    .st-emotion-cache-18tdrd9 h1 { font-size: 2.75rem; font-weight: 700; padding: 3rem 0px 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _parse_choices(choices_field):
    """Parse a choices field produced by get_table_data into a list of (label, text).
    Supports either a list of tuples, or a string like 'a-> one || b-> two'."""
    if isinstance(choices_field, list):
        return choices_field
    s = str(choices_field or "")
    parts = [p.strip() for p in s.split('||') if p.strip()]
    out = []
    for p in parts:
        if '->' in p:
            lbl, txt = [x.strip() for x in p.split('->', 1)]
            out.append((lbl, txt))
        else:
            out.append(("", p))
    return out

def _render_static_cards(table_data):
        for i, item in enumerate(table_data, start=1):
                q = item.get('MCQ', '')
                choices = _parse_choices(item.get('Choices', ''))
                correct = item.get('Correct', '')
                # build a single HTML block for the entire card to avoid stray empty blocks
                choices_html = '\n'.join([f"<div class='mcq-choice'><strong>{lbl}</strong>&nbsp;{txt}</div>" for lbl, txt in choices])
                card_html = f"""
                <div class='mcq-card'>
                    <div class='mcq-qnum'>Question {i}</div>
                    <div class='mcq-question'>{q}</div>
                    {choices_html}
                    <details style='margin-top:8px; padding-top:6px; border-top:1px dashed #e6eef9'>
                        <summary>Show answer</summary>
                        <div class='mcq-correct' style='margin-top:6px'>Answer: {correct}</div>
                    </details>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

def _render_interactive_quiz(table_data):
    # render radio buttons and allow submission/auto-grading
    st.write("### Interactive quiz")
    # create keys and radios
    for i, item in enumerate(table_data, start=1):
        q = item.get('MCQ', '')
        choices = _parse_choices(item.get('Choices', ''))
        options = [f"{lbl}) {txt}" if lbl else txt for lbl, txt in choices]
        key = f"mcq_ans_{i}"
        # set a default empty selection if not present
        if key not in st.session_state:
            st.session_state[key] = None
        st.markdown(f"**{i}. {q}**")
        st.radio(label=f"", options=options, key=key, index=0)

    if st.button("Submit Answers"):
        score = 0
        total = len(table_data)
        for i, item in enumerate(table_data, start=1):
            key = f"mcq_ans_{i}"
            selected = st.session_state.get(key)
            choices = _parse_choices(item.get('Choices', ''))
            correct_label = str(item.get('Correct', '')).strip()
            # find the canonical correct option text
            correct_text = None
            for lbl, txt in choices:
                if lbl and lbl.strip().lower() == correct_label.lower():
                    correct_text = f"{lbl}) {txt}"
                    break
                if txt.strip().lower() == correct_label.lower():
                    correct_text = f"{lbl}) {txt}"
                    break
            if selected and correct_text and selected.strip() == correct_text.strip():
                score += 1

        st.success(f"You scored {score} / {total}")
        # show per-question answers
        for i, item in enumerate(table_data, start=1):
            key = f"mcq_ans_{i}"
            selected = st.session_state.get(key)
            choices = _parse_choices(item.get('Choices', ''))
            correct_label = str(item.get('Correct', '')).strip()
            correct_text = ''
            for lbl, txt in choices:
                if lbl and lbl.strip().lower() == correct_label.lower():
                    correct_text = f"{lbl}) {txt}"
                    break
            st.markdown(f"**{i}. {item.get('MCQ','')}**")
            if selected and selected.strip() == correct_text.strip():
                st.markdown(f"<div class='mcq-correct'>Your answer: {selected} — Correct</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='mcq-wrong'>Your answer: {selected or 'No answer'} — Correct: {correct_text}</div>", unsafe_allow_html=True)



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

                        # store result in session state and render outside the form
                        if table_data and isinstance(table_data, list):
                            # remove empty questions
                            filtered = [t for t in table_data if (t.get('MCQ') or '').strip()]
                            st.session_state['last_table_data'] = filtered
                            st.session_state['show_mcqs'] = True
                        else:
                            st.session_state['last_table_data'] = None
                            st.session_state['show_mcqs'] = False
                            st.error("Unable to parse quiz into table rows.")

                else:
                    # unexpected non-dict response from chain; do not display provider output in UI
                    st.error("Generation completed but returned unexpected response format.")

# Render generated MCQs outside the form so the form doesn't expand
if st.session_state.get('show_mcqs'):
    table_data = st.session_state.get('last_table_data', [])
    st.markdown("<div class='mcq-section'>", unsafe_allow_html=True)
    try:
        _render_static_cards(table_data)
        if st.button("Take interactive quiz", key='take_quiz_outside'):
            _render_interactive_quiz(table_data)
    except Exception as e:
        st.error(f"Failed to render MCQs: {e}")
    st.markdown("</div>", unsafe_allow_html=True)