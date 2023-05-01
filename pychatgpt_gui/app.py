#! /usr/bin/env python

"""
@author: Ajay Arunachalam
Created on: 01/05/2023
Aim: A Simple Python GUI Wrapper built for unleashing the power of GPT
Version: 0.0.1
"""

from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import openai
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Iterable
import sys
import os
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import whisper as whisper
from TTS.api import TTS
from gtts import gTTS
import speech_recognition as sr
import sys
import playsound
from typing import Text
from transformers import pipeline
import utils
from api import AutoAPI, get_openai_api_key
import shutil
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from nomic.gpt4all import GPT4All
from pygpt4all.models.gpt4all import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

with open('../api_key.json', 'r') as f:
    openai_key = json.load(f)
    print(openai_key)
    
    if 'api_key' in openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key['api_key']
        
class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="repeating-linear-gradient(45deg, *primary_200, *primary_200 10px, *primary_50 10px, *primary_50 20px)",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )


seafoam = Seafoam()

# APP-1
'''Train ChatGPT on Custom Data'''

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface1 = gr.Interface(theme=seafoam, fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-data trained CHATGPT AI Bot",
                      css="footer {visibility: hidden}")

index = construct_index("../docs")
#iface1.launch(share=True)

# APP-2

'''Using openai directly'''

def analyze_text(input_text):
    # Classify the sentiment of the input text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Classify the Sentiment of this user inputted text: '{input_text}'",
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    sentiment = response.choices[0].text.strip()
    return sentiment
    


iface2 = gr.Interface(theme=seafoam, fn=analyze_text, inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Analyze Sentiment",
                      css="footer {visibility: hidden}")
#iface2.launch(share=True)

# APP - 3     
'''Using openai directly'''

def classify(input_text):
    # Classify the label of the input text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Predict the Label of this user inputted text from these provided class labels, i.e.., business, entertainment, politics, sports, technology : '{input_text}'",
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    pr_label = response.choices[0].text.strip()
    return pr_label

iface3 = gr.Interface(theme=seafoam, fn=classify, inputs=gr.components.Textbox(lines=10, label="Enter your text"),
                     outputs="text",
                     title="Predict label of your inputted text from the predefined tags based on ChatGPT",
                      css="footer {visibility: hidden}")

#iface3.launch(share=True)

# APP - 4

'''Voice Assistant with CHATGPT AI'''

def setup_voice_assistant():
    # Initialize the recognizer
    r = sr.Recognizer()
    return r

def speak_chatgpt_response_text(output_text):
    # Initialize gTTS engine
    lang_accent = 'com.en'
    filename = "user_input.wav"
    tts = gTTS(output_text, tld=lang_accent)
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)
    return tts

def ask_chatgpt(inputted_voice_text:str)->Text:
    engine = "text-davinci-003"
    results = []
    # Generate a streamed response
    for resp in openai.Completion.create(engine=engine, prompt=inputted_voice_text, max_tokens=512, n=1, stop=None, temperature=0.7, stream=True, ):
        text = resp.choices[0].text
        results.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()

    return "".join(results)
    
   
def run_voice_bot(user_input):
    
    while True:
        # Exception handling to handle exceptions at runtime if
        # no user prompt given
        try:
            r = setup_voice_assistant()
            # use the microphone as source for input.
            with sr.Microphone() as sr_audio:
                print("Microphone is open now say your prompt...")
                # wait for a second to let the recognizer
                # adjust the energy threshold cbased on
                # the surrounding noise level
                r.adjust_for_ambient_noise(sr_audio, duration=0.2)

                # listens for the user's input
                audio_user = r.listen(sr_audio)

                # Using google to recognize audio
                my_prompt = r.recognize_google(audio_user)
                my_prompt = my_prompt.lower()

                print("Did you speak :", my_prompt)
                prompt_resp_text = ask_chatgpt(my_prompt)
                speech_output = speak_chatgpt_response_text(prompt_resp_text)

        except Exception as e:
            print(e)
            print("Could not request results; {0}".format(e))
    return prompt_resp_text, speech_output

iface4 = gr.Interface(fn = run_voice_bot,
                     inputs = gr.components.Audio(source="microphone", type="numpy", label="Speak here..."),
                     outputs = ['text','audio'],
                     verbose = True,
                     title = 'ChatGPT-based Voice Assistant Bot',
                     description = 'A simple application to speak with CHATGPT Bot',
                     css="footer {visibility: hidden}"
                   )
#iface4.launch(share=True)
                   
# APP - 5

'''AutoGPT Demo APP'''

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(FILE_DIR), "auto_gpt_workspace")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

CSS = """
#chatbot {font-family: monospace;}
#files .generating {display: none;}
#files .min {min-height: 0px;}
"""

with gr.Blocks(css=CSS) as iface5:
    with gr.Column() as setup_pane:
        gr.Markdown(f"""# Auto-GPT
        1. Duplicate this Space: <a href="https://huggingface.co/spaces/{os.getenv('SPACE_ID')}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a> This will **NOT** work without duplication!
        2. Enter your <a href="https://platform.openai.com/account/api-keys">OpenAI API Key</a> below.
        """)
        with gr.Row():
            open_ai_key = gr.Textbox(
                value=get_openai_api_key(),
                label="OpenAI API Key",
                type="password",
            )
        gr.Markdown(
            "3. Fill the values below, then click 'Start'. There are example values you can load at the bottom of this page."
        )
        with gr.Row():
            ai_name = gr.Textbox(label="AI Name", placeholder="e.g. Entrepreneur-GPT")
            ai_role = gr.Textbox(
                label="AI Role",
                placeholder="e.g. an AI designed to autonomously develop and run businesses with the sole goal of increasing your net worth.",
                lines=10,
            )
        top_5_goals = gr.Dataframe(
            row_count=(5, "fixed"),
            col_count=(1, "fixed"),
            headers=["AI Goals - Enter up to 5"],
            type="array"
        )
        start_btn = gr.Button("Start", variant="primary")
        with open(os.path.join(FILE_DIR, "examples.json"), "r") as f:
            example_values = json.load(f)
        gr.Examples(
            example_values,
            [ai_name, ai_role, top_5_goals],
        )
    with gr.Column(visible=False) as main_pane:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(elem_id="chatbot")
                with gr.Row():
                    yes_btn = gr.Button("Yes", variant="primary", interactive=False)
                    consecutive_yes = gr.Slider(
                        1, 10, 1, step=1, label="Consecutive Yes", interactive=False
                    )
                custom_response = gr.Textbox(
                    label="Custom Response",
                    placeholder="Press 'Enter' to Submit.",
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.HTML(
                    lambda: f"""
                        Generated Files
                        <pre><code style='overflow-x: auto'>{utils.format_directory(OUTPUT_DIR)}</pre></code>
                """, every=3, elem_id="files"
                )
                download_btn = gr.Button("Download All Files")

    chat_history = gr.State([[None, None]])
    api = gr.State(None)

    def start(open_ai_key, ai_name, ai_role, top_5_goals):
        auto_api = AutoAPI(open_ai_key, ai_name, ai_role, top_5_goals)
        return gr.Column.update(visible=False), gr.Column.update(visible=True), auto_api

    def bot_response(chat, api):
        messages = []
        for message in api.get_chatbot_response():
            messages.append(message)
            chat[-1][1] = "\n".join(messages) + "..."
            yield chat
        chat[-1][1] = "\n".join(messages)
        yield chat

    def send_message(count, chat, api, message="Y"):
        if message != "Y":
            count = 1
        for i in range(count):
            chat.append([message, None])
            yield chat, count - i
            api.send_message(message)
            for updated_chat in bot_response(chat, api):
                yield updated_chat, count - i

    def activate_inputs():
        return {
            yes_btn: gr.Button.update(interactive=True),
            consecutive_yes: gr.Slider.update(interactive=True),
            custom_response: gr.Textbox.update(interactive=True),
        }

    def deactivate_inputs():
        return {
            yes_btn: gr.Button.update(interactive=False),
            consecutive_yes: gr.Slider.update(interactive=False),
            custom_response: gr.Textbox.update(interactive=False),
        }

    start_btn.click(
        start,
        [open_ai_key, ai_name, ai_role, top_5_goals],
        [setup_pane, main_pane, api],
    ).then(bot_response, [chat_history, api], chatbot).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )

    yes_btn.click(
        deactivate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    ).then(
        send_message, [consecutive_yes, chat_history, api], [chatbot, consecutive_yes]
    ).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )
    custom_response.submit(
        deactivate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    ).then(
        send_message,
        [consecutive_yes, chat_history, api, custom_response],
        [chatbot, consecutive_yes],
    ).then(
        activate_inputs, None, [yes_btn, consecutive_yes, custom_response]
    )

    def download_all_files():
        shutil.make_archive("outputs", "zip", OUTPUT_DIR)

    download_btn.click(download_all_files).then(None, _js=utils.DOWNLOAD_OUTPUTS_JS)

iface5.queue(concurrency_count=20)

# APP-6
'''Q&A bot with Conversational Retrieval Chain for your Custom-data'''


os.environ["OPENAI_API_KEY"] = openai_key['api_key']
llm = ChatOpenAI(temperature=0,model_name="gpt-4")

# Data Ingestion
pdf_loader = DirectoryLoader('../qa_docs/', glob="**/*.pdf")
excel_loader = DirectoryLoader('../qa_docs/', glob="**/*.txt")
word_loader = DirectoryLoader('../qa_docs/', glob="**/*.docx")
logs_loader = DirectoryLoader('../qa_docs/', glob="**/*.txt")

loaders = [pdf_loader, excel_loader, word_loader, logs_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Chunk and Embeddings
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Initialise Langchain - Conversation Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())

with gr.Blocks() as iface6:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []
    
    def user(user_message, history):
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": history})
        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        return gr.update(value=""), history
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)
    
iface6.queue(concurrency_count=20)


# APP-7
'''Using open-source GPT4 large language model based on LLaMa and GPT-J'''

def new_text_callback(text):
    print(text, end="")

def run_llm(prompt_input):
    from pygpt4all.models.gpt4all import GPT4All
    obj = GPT4All('../models/ggml-gpt4all-l13b-snoozy.bin') #'../models/ggml-gpt4all-l13b-snoozy.bin'
    resp = obj.generate(f"'{prompt_input}'", n_predict=100, new_text_callback=new_text_callback)
    return resp

iface7 = gr.Interface(theme=seafoam, fn=run_llm, inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs=gr.components.Textbox(lines=10),
                     title="Open-Source LLM Demo",
                      css="footer {visibility: hidden}")

#iface7.launch(share=True)


# APP - 8
'''Using LangChain to interact with GPT4All models'''

# Download a GPT4All model from http://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin

# Download the GPT4All-J model from https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin

def run_langchain_gpt4(prompt_input, local_path='../models/ggml-gpt4all-l13b-snoozy.bin'):
    from langchain.llms import GPT4All
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callback_manager=callback_manager, verbose=True) #
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question = f"'{prompt_input}'"
    response = llm_chain.run(question)
    return response

iface8 = gr.Interface(theme=seafoam, fn=run_langchain_gpt4, inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs=gr.components.Textbox(lines=10),
                     title="GPT4All Model Demo",
                      css="footer {visibility: hidden}")
                      
#iface8.launch(share=True)


# combine to create a multipage app
chatgpt_custom_ui_demo = gr.TabbedInterface([iface1, iface2, iface3, iface4, iface5, iface6, iface7, iface8], ["Train ChatGPT Model on Your Own Documents", "Sentiment Analyzer with ChatGPT", "Predict Input Label using Predefined Engineered Prompt - ChatGPT", "Voice Assistant with CHATGPT AI","AUTOGPT DEMO", "Conversational Retrieval QA ChainBot","Open-Source LLM","GPT4All"], theme=seafoam, css="footer {visibility: hidden}") 

if __name__ == "__main__":
    chatgpt_custom_ui_demo.launch(debug=True, share=True, file_directories=[OUTPUT_DIR])
    chatgpt_custom_ui_demo.queue(concurrency_count=20)