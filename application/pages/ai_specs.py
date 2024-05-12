import streamlit as st
from utils import *
import json

# optional variation of all nn specs used.
#:"temperature": 0.7, "max_length": 60, "max_new_tokens":30, "top_k": 60, "repetition_penalty": 1.1 model 1
# model 1 and model 3

st.set_page_config("ai_specs")
st.title("Try out different Parameters or create your own model")
st.error("this page is currently under construction")

MODELS = Path("my_models")
if not MODELS.exists():
    MODELS.mkdir(exist_ok=True, parents=True)

create_model_form = st.form(key = "model_creator")

def output_dimensions(x, padding, kernel_size, dilation, stride):
        return int((x * 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def save_custom_model(size_of_input, neurons, num_of_conv_layers, kernel_sizes, strides):
    input_size = size_of_input
    for layer in range(num_of_conv_layers - 1):
        if (input_size := output_dimensions(input_size, 0, kernel_sizes[layer], 1, strides[layer])) < kernel_sizes[layer + 1] or input_size < 1:
            print(input_size)
            st.error("The chosen parameters are not valid. Please decrease the number of layers, the kernel sizes, the stride values or increase the input size")
            return 

    model_parameters = {"input": size_of_input, "neurons": neurons, "num_of_conv_layers": num_of_conv_layers, "kernel_sizes": kernel_sizes, "strides": strides}

    with open(MODELS / f"model_{int(len(os.listdir(MODELS))) + 1}.json", "w") as f:
        json.dump(model_parameters, f)


with  create_model_form:
    st.header("here you can create your own model to sort out irrelevant pages")

    size_of_input = st.select_slider("size of the input images", [64, 128, 256, 512])
    neurons = st.slider("neurons per layer", 10, 300, 120, 1)
    num_of_conv_layers = st.slider("number of convoluitonal layers", 1, 10, 3, 1)
    kernel_sizes = [st.slider(f"kernel size of layer {i}", 1, 5, 2) for i in range(num_of_conv_layers)]
    strides = [st.slider(f"stride of layer {i}", 1, 4, 1) for i in range(num_of_conv_layers)]

    if st.form_submit_button("submit"):
        pass

if st.button("save model"):
    save_custom_model(size_of_input, neurons, num_of_conv_layers, kernel_sizes, strides)


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