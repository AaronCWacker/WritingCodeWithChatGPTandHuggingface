# Writing Python With ChatGPT and Huggingface

This is a quick demonstration to get anyone up to speed with generating AI and Python programs using state of the art AI.

Our first example is ChatGPT.  Here is a step by step to get started:

1. Go here: https://chat.openai.com/chat
2. If that was down, try here:  https://platform.openai.com/playground
3. Register for the research preview, gain access then return to URL.
4. Optionally there is a discord link from a prompt engineering cchannel here: https://discord.com/channels/974519864045756446/1046317269069864970

Once logged in:

Here is an example of a session with prompt in and output out.

I'm having fun doing small directed python programs where the solution is tweaked from prompt to prompt with intuition about which library to use.  This produces something really fundamentally useful for quick AI program development.  

# For instance:  
"write a python program that creates a pandas dataframe from a CSV file then creates dynamic filters one for each field." 

# Yields: 

```
"import pandas as pd

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
```

Zooming in to a better IO sdk I choose streamlit.  

Same with streamlit as the variation on the theme: 

# Prompt 2:

"write a python program using the streamlit library that creates a pandas dataframe from a CSV file then creates dynamic filters one for each field."  which not only generated the program but annotated how to run it as a streamlit app: 

```
"import pandas as pd
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
```


The resulting program worked after one modification uploading a file to Huggingface spaces and running as streamlit app.  This took less than 5 minutes:

![image](https://user-images.githubusercontent.com/30595158/217809016-fe695232-8339-4a98-b966-39c3568be619.png)


Cites and References:

https://huggingface.co/awacke1

Me - Human - DNA model below ;) :

![image](https://user-images.githubusercontent.com/30595158/217809156-6b7f0be5-b645-4076-b567-89b3c71a1546.png)


# Generated App:  

https://huggingface.co/spaces/awacke1/PandasDataframeAutoFilterStreamlit

ChatbotMemory.csv Dataset: 
https://huggingface.co/datasets/awacke1/ChatbotMemory.csv/blob/main/ChatbotMemory.csv

Wikipedia AI Search: Use to fact check:  
https://huggingface.co/spaces/awacke1/WikipediaUltimateAISearch

![image](https://user-images.githubusercontent.com/30595158/217810205-0e20d2ad-826a-4b19-b5eb-f49ca23db65a.png)


# Gradio program generation example:

Prompt:
create a python program that uses the gradio library to generate an interface which loads three huggingface spaces and displays them in the same gradio program.

Output Code:
```
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
```

# Defects?  Test
1. gr.interface.Radio is incorrect.  When using gradio interface elements, use gr.Radio without interface.
2. gr.inputs.Textbox - same
3. Notice it created three references to models rather than spaces like I asked?  Its a stronger pattern statiscally but the embed multiple spaces pattern may have been created after the cut date of last training data which was some time in 2021..
4. There was an assumption that we didn't need to load requirements.txt - pip installing references to sdks and libraries.

Add a requirements.txt file with the two imports:

1. transformers
2. torch

Correct one liner here.  The case for the class Interface was lower case which is incorrect.  Python is case sensitive and the Interface class is as follows:
models = gr.Interface.Radio(["BERT", "RoBERTa", "DistilBERT"], default="BERT")

# Complex example with specific AI UI dataset loading UI components and graph components

# Prompt:
write a python program that uses the pandas CSV file loader to load a file, then create a gradio application that uses the fields to create a filter for each and use the gradio Interface the Dataframe, the UploadButton, and the Plot to automatically shows a plotly graph object with the results.  Also create a gradio.Examples input for three files that are CSV files.

# Answer:

```
import gradio as gr
import pandas as pd
import plotly.express as px

# Define the function to load the CSV file
def load_data(file):
    df = pd.read_csv(file)
    return df

# Create the input interface for the CSV file
inputs = gr.inputs.UploadButton(label="Upload your CSV file:")

# Define the main function that will be called when the user submits their file
def main(inputs):
    df = load_data(inputs)
    return df

# Create the output interface for the DataFrame
outputs = gr.outputs.Dataframe(examples={"file1.csv": "path/to/file1.csv",
                                          "file2.csv": "path/to/file2.csv",
                                          "file3.csv": "path/to/file3.csv"})

# Create the Plot output interface
plot = gr.outputs.Plotly(click_submit_button=True)

# Define the plot function that will be called when the user selects their options
def plot_function(df, x, y):
    fig = px.scatter(df, x=x, y=y)
    return fig

# Create the Gradio interface
interface = gr.Interface(fn=main, inputs=inputs, outputs=outputs, title="CSV Plotter", plot=plot, plot_fn=plot_function)
interface.launch()
```

# Prompt 2:

Write a python program using gradio that writes a CSV file with three columns: RecordCount, Topic, Intervention and output are the three field names, writes the file and then loads the file and with three example records draws a plotly Plot with the three values.


# Output 2:

```
import gradio as gr
import pandas as pd
import plotly.express as px

# Create a function to write the CSV file
def write_csv(record_count, topic, intervention):
    df = pd.DataFrame({
        "RecordCount": [record_count],
        "Topic": [topic],
        "Intervention": [intervention]
    })
    df.to_csv("records.csv", index=False, mode='a')

# Create a function to plot the data from the CSV file
def plot_data():
    df = pd.read_csv("records.csv")
    fig = px.scatter(df, x="RecordCount", y="Intervention", color="Topic")
    return fig

# Define the inputs for the Gradio interface
inputs = [
    gr.inputs.Slider(label="Record Count", min=0, max=100, default=50),
    gr.inputs.Textbox(label="Topic"),
    gr.inputs.Textbox(label="Intervention")
]

# Define the outputs for the Gradio interface
outputs = [
    gr.outputs.Plotly(plot_data),
    gr.outputs.Textbox(label="Data written to records.csv")
]

# Create the Gradio interface
interface = gr.Interface(write_csv, inputs, outputs, title="Record Plotter")

# Launch the Gradio interface
interface.launch()

```


The description of the program was also sublime:

```
This program creates a Gradio interface with three inputs: Record Count, Topic, and Intervention. When the user inputs the values and submits the form, the values are passed to the write_csv function, which writes the data to a CSV file named records.csv. The plot_data function is then called to read the CSV file and create a scatter plot using Plotly. The plot and a text message indicating that the data was written to the CSV file are then displayed in the Gradio interface.
```


# Test and Analyze Your New Program: What Works?  What Does Not Work?
https://huggingface.co/spaces/awacke1/GradioAutoCSVLoaderToPlotly

# Defects:
```
Traceback (most recent call last):
  File "app.py", line 22, in <module>
    gr.inputs.Slider(label="Record Count", min=0, max=100, default=50),
TypeError: __init__() got an unexpected keyword argument 'min'
```
    
# Fixes:
```
gr.inputs.Slider(label="Record Count", minimum=0, maximum=100, default=50),

    
# Define the outputs for the Gradio interface
outputs = [
    gr.Plot(plot_data),
    gr.Textbox(label="Data written to records.csv")
]
```

# Program that plots via a CSV file and can write new record to CSV is here: 

https://huggingface.co/spaces/awacke1/GradioAutoCSVLoaderToPlotly

![image](https://user-images.githubusercontent.com/30595158/217847864-c9f8cfcb-94ab-47d1-bacb-fd91fbaff8c6.png)

testfile.csv is here:  https://huggingface.co/spaces/awacke1/GradioAutoCSVLoaderToPlotly/blob/main/testfile.csv




# Try again in streamlit

```
Write a python program using streamlit that writes a CSV file with three columns: RecordCount, Topic, Intervention and output are the three field names, writes the file and then loads the file and with three example records draws a plotly Plot with the three values.
```

```
import streamlit as st
import pandas as pd
import plotly.express as px

# Create a function to write the CSV file
def write_csv(record_count, topic, intervention):
    df = pd.DataFrame({
        "RecordCount": [record_count],
        "Topic": [topic],
        "Intervention": [intervention]
    })
    df.to_csv("records.csv", index=False, mode='a')

# Create a function to plot the data from the CSV file
def plot_data():
    df = pd.read_csv("records.csv")
    fig = px.scatter(df, x="RecordCount", y="Intervention", color="Topic")
    return fig

st.set_page_config(page_title="Record Plotter", page_icon=":chart_with_upwards_trend:", layout="wide")

# Get the input values from the user
record_count = st.sidebar.slider("Record Count", min_value=0, max_value=100, value=50)
topic = st.sidebar.text_input("Topic")
intervention = st.sidebar.text_input("Intervention")

# Write the data to the CSV file
write_csv(record_count, topic, intervention)

# Plot the data from the CSV file
fig = plot_data()
st.plotly_chart(fig)

# Display a message indicating that the data was written to the CSV file
st.success("Data written to records.csv")
```
