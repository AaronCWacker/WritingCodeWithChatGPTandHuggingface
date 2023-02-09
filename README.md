# WritingCodeWithChatGPTandHuggingface

This is a quick demonstration to get anyone up to speed with generating AI and Python programs using state of the art AI.

Our first example is ChatGPT.  Here is a step by step to get started:

1. Go here: https://chat.openai.com/chat
2. Register for the research preview, gain access then return to URL.
3. Optionally there is a discord link from a prompt engineering cchannel here: https://discord.com/channels/974519864045756446/1046317269069864970

Once logged in:

Here is an example of a session with prompt in and output out.
I'm having fun doing small directed python programs where the solution is tweaked from prompt to prompt with intuition about which library to use.  This produces something really fundamentally useful for quick AI program development.  For instance:  "write a python program that creates a pandas dataframe from a CSV file then creates dynamic filters one for each field." yields: "import pandas as pd

def create_dataframe(file_path):
    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(file_path)

    # Create dynamic filters for each field
    filters = {}
    for col in df.columns:
        filters[col] = df[col].unique().tolist()

    return df, filters

if name == 'main':
    file_path = 'path/to/csv/file.csv'
    df, filters = create_dataframe(file_path)
    print('Dataframe:')
    print(df)
    print('\nFilters:')
    print(filters)
"

Zooming in to a better IO sdk I choose streamlit.  Same with streamlit as the variation on the theme: "write a python program using the streamlit library that creates a pandas dataframe from a CSV file then creates dynamic filters one for each field."  which not only generated the program but annotated how to run it as a streamlit app: "import pandas as pd
import streamlit as st

def create_dataframe(file_path):
    # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(file_path)

    # Create dynamic filters for each field
    filters = {}
    for col in df.columns:
        filters[col] = df[col].unique().tolist()

    return df, filters

if name == 'main':
    file_path = 'path/to/csv/file.csv'
    df, filters = create_dataframe(file_path)

    # Use Streamlit to display the dataframe and filters
    st.write('Dataframe:', df)
    st.write('Filters:', filters)
#In this code, we first read the CSV file into a Pandas dataframe using the pd.read_csv function. Then, we create a dictionary filters to store the unique values of each field in the dataframe. Finally, we use the streamlit library to display both the dataframe and the filters in a web-based interface.

#You can run this code in your terminal or command prompt by typing streamlit run filename.py, where filename.py is the name of the file containing this code. The output will be a web-based interface showing the dataframe and the filters, which you can interact with and explore.


"

The resulting program worked after one modification uploading a file to Huggingface spaces and running as streamlit app.  This took less than 5 minutes:

![image](https://user-images.githubusercontent.com/30595158/217809016-fe695232-8339-4a98-b966-39c3568be619.png)


Cites and References:

https://huggingface.co/awacke1

DNA:

![image](https://user-images.githubusercontent.com/30595158/217809156-6b7f0be5-b645-4076-b567-89b3c71a1546.png)


Generated App:  https://huggingface.co/spaces/awacke1/PandasDataframeAutoFilterStreamlit

ChatbotMemory.csv Dataset: https://huggingface.co/datasets/awacke1/ChatbotMemory.csv/blob/main/ChatbotMemory.csv
Wikipedia AI Search: Use to fact check:  https://huggingface.co/spaces/awacke1/WikipediaUltimateAISearch

![image](https://user-images.githubusercontent.com/30595158/217810205-0e20d2ad-826a-4b19-b5eb-f49ca23db65a.png)


# Gradio program generation example:

Prompt:
create a python program that uses the gradio library to generate an interface which loads three huggingface spaces and displays them in the same gradio program.

Output Code:
'''
import gradio as gr
import transformers

# Load the three models using Hugging Face's transformers library
model1 = transformers.AutoModel.from_pretrained("bert-base-cased")
model2 = transformers.AutoModel.from_pretrained("roberta-base")
model3 = transformers.AutoModel.from_pretrained("distilbert-base-cased")

# Define a function that takes an input text and returns the model's output
def get_output(model, input_text):
    input_ids = transformers.preprocessing.tokenization_utils.map_text_to_ids(tokenizer, input_text)
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    output = model(input_tensor)[0]
    return output.tolist()[0]

# Create an input interface for the user to enter their text
inputs = gr.inputs.Textbox(lines=2, label="Enter your text here:")

# Create a radio button interface to select which model to use
models = gr.interface.Radio(["BERT", "RoBERTa", "DistilBERT"], default="BERT")

# Define the main function that will be called when the user submits their text and model choice
def main(inputs, models):
    input_text = inputs
    if models == "BERT":
        output = get_output(model1, input_text)
    elif models == "RoBERTa":
        output = get_output(model2, input_text)
    else:
        output = get_output(model3, input_text)
    return output

# Create and launch the Gradio interface
interface = gr.Interface(fn=main, inputs=inputs, interfaces=models, out_features=gr.outputs.Textbox(lines=10), title="Hugging Face Models")
interface.launch()
'''



