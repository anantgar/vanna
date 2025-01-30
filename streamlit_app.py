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
        else:
            st.code(msg["content"], language="sql")

# Handle user input
if user_input := st.chat_input():
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
        else:
            st.session_state.messages.append({"role": "assistant", "content": "No data was returned for your query."})
            with st.chat_message("assistant"):
                st.write("No data was returned for your query.")
        
        summary = vn.generate_summary(user_input, df)
        st.session_state.messages.append({"type": "summary", "role": "assistant", "content": summary})
        with st.chat_message("assistant"):
            st.write(summary)
    except Exception as e:
        st.exception(e)