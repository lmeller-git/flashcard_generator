from dotenv import load_dotenv
from torch import nn 
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from utils_for_models import load_state_dict_2
import langchain.globals
from langchain.cache import InMemoryCache

load_dotenv()
langchain.globals.set_debug(False)
langchain.globals.set_llm_cache(InMemoryCache())
langchain.globals.set_verbose(False)
repo_id = "google/flan-t5-xxl"
repo_id_2 = "Falconsai/medical_summarization"
repo_id_3 = "mistralai/Mixtral-8x7B-v0.1"

class CustomPageDecider(nn.Module):
    
    def __init__(self, neurons, kernels, num_layers, strides, image_size, input_size = 3, output_size = 2):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=neurons, kernel_size=kernels[0], stride=strides[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=neurons,
                      out_channels=neurons, kernel_size=kernels[0], stride=strides[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_layers = [nn.Sequential(nn.Conv2d(in_channels=neurons,
                        out_channels=neurons, kernel_size=kernels[i + 1], stride=strides[i + 1]),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=neurons,
                                out_channels=neurons, kernel_size=kernels[i + 1], stride=strides[i + 1]),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=kernels[i + 1])) for i in range(num_layers - 1)]

        height = image_size

        for i in range(num_layers):
            height = self.output_dimensions(height, 0, kernels[i], 1, strides[i])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=height,
                      out_features=output_size)
        )

    def forward(self, x):
        x = self.layer_1(x)
        for i in range(len(self.conv_layers)):
            print(x.shape)
            x = self.conv_layers[i](x)
        print(x.shape)
        return self.classifier(x)

    def output_dimensions(self, x, padding, kernel_size, dilation, stride):
        return int((x * 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

class PageDecider(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, x_kernel, y_kernel): #hidden layer size can vary for different models
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_dim, kernel_size=(x_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(x_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(x_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(x_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(y_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(y_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(x_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(y_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(y_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(x_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(y_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(y_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim, kernel_size=(x_kernel,y_kernel), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_dim * 9,
                      out_features=output_size)
        )
        
    def forward(self, x):
        return self.classifier(self.layer_3(self.layer_2(self.layer_1(x))))

model_decide = PageDecider(3, 120, 2, 3, 3)

model_decide = load_state_dict_2(model_decide, "models/page_decider_2.pth")

model = HuggingFaceHub(repo_id = repo_id, model_kwargs = {"temperature": 0.7, "max_length": 60, "max_new_tokens":30, "top_k": 60, "repetition_penalty": 1.1}) # summary
model_2 = HuggingFaceHub(repo_id = repo_id_2, model_kwargs = {"temperature": 0.6, "max_length": 40}) # none
model_3 = HuggingFaceHub(repo_id = repo_id, model_kwargs = {"temperature": 0.4, "max_length": 10}) # questions


prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a llm, which condenses a given input into a short summary, suitable for a flashcard"),
    ("user", "{input}")
])

prompt_2 = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

prompt_3 = ChatPromptTemplate.from_messages([
     ("system", "you are a llm, which provides a short question, suitable for a flashcard to a given input"),
     ("user", "{input}")
])

parser = StrOutputParser()
hub_chain = LLMChain(prompt=prompt, llm=model) 
hub_chain_2 = LLMChain(prompt = prompt_2, llm=model_2)
hub_chain_3 = LLMChain(prompt=prompt_3, llm=model_3)
