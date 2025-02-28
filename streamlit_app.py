import streamlit as st
import sys
sys.path.insert(0, "libs")  # Prioritize your local version
# import vanna
# print("Loaded vanna from:", vanna.__file__)  
import base64
import os
import psycopg2
from vanna.openai import OpenAI_Chat
from vanna.vannadb import VannaDB_VectorStore
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

st.set_page_config(page_title="Chat with Your Data", layout="wide")
st.title("💬 Talk to Your Data")

openai_api_key = st.secrets["openai_api_key"]
client = OpenAI(
    api_key=openai_api_key
)

def TTS(text: str):
    response = client.audio.speech.create(
        model='tts-1',
        voice='echo',
        input=text
    )

    # Save the audio file
    audio_path = "output.mp3"
    response.write_to_file(audio_path)

    # Read audio file
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    # Convert to base64 for auto-play
    audio_base64 = base64.b64encode(audio_bytes).decode()

    # HTML + JavaScript for autoplay
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """

    # Embed in Streamlit app
    st.markdown(audio_html, unsafe_allow_html=True)
    os.remove(audio_path)

conn = psycopg2.connect(
    dbname=st.secrets["database"],
    user=st.secrets["user"],
    password=st.secrets["db_password"],
    host=st.secrets["host"],
    port=5439
)

def run_sql(sql: str) -> pd.DataFrame:
    df = pd.read_sql_query(sql, conn)
    return df

class MyVanna(VannaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        VannaDB_VectorStore.__init__(self, vanna_model=st.secrets["vanna_model"], vanna_api_key=st.secrets["vanna_api_key"], config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(
    config={'api_key': openai_api_key,
            'model': 'gpt-4o'}
)
vn.run_sql = run_sql
vn.run_sql_is_set = True

# Initialize conversation history
if "conv_hist" not in st.session_state:
    st.session_state.conv_hist = []

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"type": "opening", "role": "assistant", "content": "How can I help you with your data today?"}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], pd.DataFrame):
            st.dataframe(msg["content"], use_container_width=True)
        elif "user" in msg["role"] or "opening" in msg["type"] or "summary" in msg["type"]:
            st.write(msg["content"])
        elif msg["type"] == "plotly":
            st.plotly_chart(msg["content"])
        else:
            st.code(msg["content"], language="sql")

audio_value = st.audio_input("Press the mic button to record", key="voice_input")

if audio_value:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file = audio_value
    )
    transcript_text = transcript.text  # Extract text
    st.session_state["transcribed_text"] = transcript_text  # Store transcription
    # st.rerun()  # Refresh UI to update chat input   

user_input = st.chat_input("Type a message or use voice input")

if "transcribed_text" in st.session_state and not user_input:
    user_input = st.session_state["transcribed_text"]  # Use transcribed text as input
    del st.session_state["transcribed_text"]  # Clear stored transcription

# Handle user input
if user_input:
    # Add user message to the conversation
    st.session_state.messages.append({"type": "user", "role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate SQL and query the database
    if len(st.session_state.conv_hist) > 0:
        prompt = "Previous prompts and generated SQL queries:\n" + "\n".join(st.session_state.conv_hist) + "\nAnswer the following question based on the history: This is a Redshift database. " + user_input
    else:
        prompt = "This is a Redshift database. " + user_input
    
    sql = vn.generate_sql(prompt)
    st.session_state.conv_hist.append(user_input)
    st.session_state.messages.append({"type": "sql", "role": "assistant", "content": sql})
    st.session_state.conv_hist.append(sql)
    
    with st.chat_message("assistant"):
        st.code(sql, language="sql")

    try:
        df = vn.run_sql(sql)
        if not df.empty:
            st.session_state.messages.append({"type": "df", "role": "assistant", "content": df})
            with st.chat_message("assistant"):
                st.dataframe(df, use_container_width=True)

            plotly_gen = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer", "content": "Determine if the user's request/question explicitly asks for a chart, plot, or graph. Return True or False."},
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
            )

            if plotly_gen.choices[0].message.content == "True":
                # Generate Plotly code
                plotly_code = vn.generate_plotly_code(question=user_input, sql=sql, df_metadata=str(df.describe()))
                st.session_state.messages.append({"type": "plotly_code", "role": "assistant", "content": plotly_code})
                print(plotly_code)
                # Generate and display the Plotly figure

                plotly_figure = vn.get_plotly_figure(plotly_code, df, dark_mode=True)
                st.session_state.messages.append({"type": "plotly", "role": "assistant", "content": plotly_figure})
                with st.chat_message("assistant"):
                    st.plotly_chart(plotly_figure)

            else:
                # Generate and display summary
                summary = vn.generate_summary(user_input, df)
                TTS(summary)
                st.session_state.messages.append({"type": "summary", "role": "assistant", "content": summary})
                with st.chat_message("assistant"):
                    st.write(summary)
        else:
            st.session_state.messages.append({"type": "error", "role": "assistant", "content": "No data was returned for your query."})
            with st.chat_message("assistant"):
                st.write("No data was returned for your query.")
    except Exception as e:
        st.exception(e)