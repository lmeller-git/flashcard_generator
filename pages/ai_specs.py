import streamlit as st

# optional variation of all nn specs used.
#:"temperature": 0.7, "max_length": 60, "max_new_tokens":30, "top_k": 60, "repetition_penalty": 1.1 model 1
# model 1 and model 3

st.set_page_config("ai_specs")
st.title("Try out different Parameters")
st.error("this page is currently under construction")
summary_holder = st.container()
with summary_holder:
    sliders_summary = st.form("sliders_s")
sliders_questions = st.form("sliders_q")

def apply_parameters():
    pass

def reset_func():
    pass




with sliders_summary:
    st.header("Here you can change the parameters of the summarizer")
    temperature = st.slider(label="temperature",  min_value=0.3,  value=0.7, max_value=1.0)
    max_length = st.slider(label="max_length", min_value=10,  value=60, max_value=120)
    max_new_tokens = st.slider(label="max_new_tokens",  min_value=10,  value=30, max_value=60)
    top_k = st.slider(label="top k",  min_value=10,  value=60, max_value=120)
    repetition_penalty = st.slider(label="Repetition Penalty", min_value=1.0,  value=1.1, max_value=1.5)
    st.form_submit_button("submit changes", on_click=apply_parameters)

with sliders_questions:
    st.header("Here you can change the parameters of the question generator")
    temperature = st.slider(label="temperature",  min_value=0.3,  value=0.7, max_value=1.0)
    max_length = st.slider(label="max_length", min_value=10,  value=60, max_value=120)
    max_new_tokens = st.slider(label="max_new_tokens",  min_value=10,  value=30, max_value=60)
    top_k = st.slider(label="top k",  min_value=10,  value=60, max_value=120)
    repetition_penalty = st.slider(label="Repetition Penalty", min_value=1.0,  value=1.1, max_value=1.5)
    st.form_submit_button("submit changes", on_click=apply_parameters)

reset = st.button("reset all changes", on_click=reset_func)
if reset:
    st.rerun()