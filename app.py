import streamlit as st
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

st.set_page_config(page_title="MedExplainAI", layout="centered")

@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    return tokenizer, model

tokenizer, model = load_model()

def simplify_prompt(text):
    return f"""
Simplify the following medical text into very simple and clear language.

Rules:
- Keep the meaning correct
- Avoid medical jargon
- Keep it short and easy to understand
- Do NOT give diagnosis

Text:
{text}
"""

def qa_prompt(question):
    return f"""
Answer the following medical question in simple language.

Rules:
- Give educational information only
- Do NOT provide diagnosis
- Do NOT give treatment advice
- Keep it easy to understand
- Add a short note suggesting to consult a doctor

Question:
{question}
"""

def clean_output(result, original_prompt):
    if result.startswith(original_prompt):
        result = result[len(original_prompt):].strip()
    return result

def simplify_text(text):
    prompt = simplify_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(result, prompt)

def answer_question(question):
    prompt = qa_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(result, prompt)

st.title("MedExplainAI")
st.write("A web-based medical text simplification and medical question-answering system.")
st.warning("This system is for educational purposes only. It does not provide diagnosis or treatment. Please consult a licensed doctor for medical advice.")

mode = st.selectbox("Choose a task", ["Simplify Clinical Text", "Ask Medical Question"])
user_input = st.text_area("Enter your text or question here")

if st.button("Generate"):
    if not user_input.strip():
        st.error("Please enter some text first.")
    else:
        start_time = time.time()

        if mode == "Simplify Clinical Text":
            output = simplify_text(user_input)
        else:
            output = answer_question(user_input)

        end_time = time.time()

        st.subheader("Result")
        st.write(output)
        st.write(f"Response time: {round(end_time - start_time, 2)} seconds")