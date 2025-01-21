import streamlit as st
import redshift_connector
from vanna.openai import OpenAI_Chat
from vanna.vannadb import VannaDB_VectorStore
import pandas as pd

st.set_page_config(page_title="Chat with Your Data", layout="wide")
st.title("💬 Talk to Your Data")

openai_api_key = st.secrets["openai_api_key"]

conn = redshift_connector.connect(
    database=st.secrets["database"],
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

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with your data today?"}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], pd.DataFrame):
            st.dataframe(msg["content"], use_container_width=True)
        elif "How can I help you with your data today?" in msg["content"] or "user" in msg["role"]:
            st.write(msg["content"])
        else:
            st.code(msg["content"], language="sql")

# Handle user input
if user_input := st.chat_input():
    # Add user message to the conversation
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate SQL and query the database
    sql = vn.generate_sql(user_input)
    st.session_state.messages.append({"role": "assistant", "content": sql})
    with st.chat_message("assistant"):
        st.code(sql, language="sql")

    df = vn.run_sql(sql)

    # Add the bot's response (and optionally the DataFrame) to the conversation
    if not df.empty:
        st.session_state.messages.append({"role": "assistant", "content": df})
        with st.chat_message("assistant"):
            st.dataframe(df, use_container_width=True)
    else:
        st.session_state.messages.append({"role": "assistant", "content": "No data was returned for your query."})
        with st.chat_message("assistant"):
            st.write("No data was returned for your query.")
