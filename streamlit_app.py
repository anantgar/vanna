import streamlit as st
import redshift_connector
from vanna.openai import OpenAI_Chat
from vanna.vannadb import VannaDB_VectorStore
import pandas as pd

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

# Input field for the user to ask a question
my_question = st.text_input("Ask me a question about your data:")

# Check if the user has entered a question
if my_question:
    st.text(my_question)
    sql = vn.generate_sql(my_question)
    st.text(sql)
    df = vn.run_sql(sql)    
    st.dataframe(df, use_container_width=True)