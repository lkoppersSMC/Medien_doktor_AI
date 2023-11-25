from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI

import os
import json
from pprint import pprint

import chromadb
from chromadb.config import Settings

import pandas as pd

load_dotenv()
DATA_LOCATION = os.environ.get("DATA_LOCATION")

def _get_dbs():
    """Erzeugt ChromaDB-Client und stellt Verbindung zu zwei Collections her"""
    persistent_client = chromadb.PersistentClient(
        path=f"{DATA_LOCATION}/chroma_db", settings=Settings(allow_reset=True)
    )  # should be created once and passed around

    teaser_db = Chroma(
        client=persistent_client,
        collection_name="story_teaser",
        embedding_function=OpenAIEmbeddings(),
    )

    statement_db = Chroma(
        client=persistent_client,
        collection_name="story_statement",
        embedding_function=OpenAIEmbeddings(),
    )

    return teaser_db, statement_db, persistent_client

def _chroma_to_dataframe(chroma_results):
    """Kleiner Helfer, um aus den Ergebnissen von Anfragen gegen ChromaDB DataFrames mit aufgelösten Metadaten zu erzeugen"""
    df = pd.DataFrame()

    if isinstance(chroma_results, dict):
        df = pd.DataFrame(chroma_results)
    elif isinstance(chroma_results, list) and isinstance(chroma_results[0], Document):
        df = pd.DataFrame(map(lambda x: {"documents": x.page_content, "metadatas": x.metadata}, chroma_results))

    if not df.empty:
        metadata = pd.json_normalize(df.metadatas)
        df = pd.concat([df, metadata], axis=1).drop(columns="metadatas")

    return df

if __name__ == '__main__':
    teaser_db, statement_db, _ = _get_dbs()
    res = statement_db.similarity_search("klimawandel insekten")
    pprint(res)
