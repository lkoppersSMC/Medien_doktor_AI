import os
import json
from pprint import pprint


from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chat_models import ChatOpenAI

import chromadb
from chromadb.config import Settings
import pandas as pd

from openai import OpenAI
from pathlib import Path
from helper import _get_dbs, _chroma_to_dataframe
from graphql.request_gql import get_general_query

from typing import Any, Dict, List, Optional

load_dotenv()
DATA_LOCATION = os.environ.get("DATA_LOCATION")

import glob

GPT_MODEL = "gpt-4-1106-preview"

def _get_teaser_for_story(story_no: int) -> str:
    """Get the teaser for a given story_no."""
    where_clause: str = f"story_no: {{_eq: {story_no}}}"
    res = get_general_query(
        table_name="story_meta",
        schema_name="smc",
        return_nodes="smc_content {teaser}",
        where_clause=where_clause,
        flatten_response=True
    )
    return res[0].get("smc_content")[0].get("teaser")

def _get_content_for_story_and_expert(story_no: int, expert_name: str) -> dict:
    """Get the statements for a given story_no and (Expert) contact_id."""
    where_clause: str = f"expert_statements: {{expert_name: {{_eq: \"{expert_name}\"}}, story_no: {{_eq: {story_no}}}}}"
    res = get_general_query(
        table_name="story_meta",
        schema_name="smc",
        return_nodes="""
            story_no, 
            title,  
            expert_statements {
                expert_name, 
                statement,
                question
            },
            smc_content {
                teaser
            }""",
        where_clause=where_clause,
        args_clause="order_by: {publication_date: desc}",
        flatten_response=False
    )
    return res

def _get_statements_for_story_id(story_no: int) -> dict:
    """Get the statements for a given story_no."""
    where_clause: str = f"story_no: {{_eq: {story_no}}}"
    res = get_general_query(
        table_name="story_meta",
        schema_name="smc",
        return_nodes="""
            story_no, 
            title,  
            expert_statements {
                expert_name, 
                statement,
                question
            },
        """,
        where_clause=where_clause,
        args_clause="order_by: {publication_date: desc}",
        flatten_response=False
    )
    return res


def teaser_matches_crit(teaser : str, prompt : str) -> str:
    teaser_crit_prompt = PromptTemplate(template=prompt, input_variables=["TEXT"])
    teaser_chain = teaser_crit_prompt | ChatOpenAI(temperature=0, model=GPT_MODEL) | StrOutputParser()

    teaser_match_output = teaser_chain.invoke({"TEXT": teaser})
    return teaser_match_output


def gen_sum_statements(story_no: int) -> List:
    possible_candidates_teaser: list = []
    possible_candidates_statements: list = []
    possible_candidates_experts: list = []

    debug_output: Dict[str, Any] = {}

    
    teaser_db, statement_db, _ = _get_dbs()

    story_no: story_no
    
    
    statements = _get_statements_for_story_id(story_no)
    for statement in statements.get("data").get("smc_story_meta")[0].get("expert_statements"):
        possible_candidates_statements.append(statement.get("statement"))
        possible_candidates_experts.append(statement.get("expert_name"))
        debug_output[statement.get("expert_name")] = statement.get("statement")

    #****
    # Text Processing - Summery
    #****

    pprint(len(possible_candidates_teaser))

    debug_output["possible_candidates_teaser"] = possible_candidates_teaser

    text_chain_prompt_template = """
    You are an expert in summerizing text. You will receive a text and have to summerize it less than 5 sentences. You will also 
    receive a CONTEXT, which you should use to summerize the text. The context is the following:```

    CONTEXT: {CONTEXT}
    ```
    TEXT: {TEXT}
    ```
    
    keep the context in mind when summerizing the text.
    Please answer in German.
    """

    text_chain_prompt = PromptTemplate(template=text_chain_prompt_template, input_variables=["CONTEXT", "TEXT"])
    text_chain = text_chain_prompt | ChatOpenAI(temperature=0, model=GPT_MODEL) | StrOutputParser()

    summeraized_statements = [text_chain.invoke({"CONTEXT": "", "TEXT": statement}) for statement in possible_candidates_statements]

    debug_output["summeraized_statements"] = summeraized_statements
    return summeraized_statements

def gen_sum_statements_sums(summeraized_statements):
    statement_chain_prompt_template = """
    You are an expert in summerizing text. You will receive list of Text Snippets and have to summerize it less than 5 sentences.
    The Text is formatted as follows:
    <statement_begin>TEXT<statement_end>

    ```
    TEXT: {TEXT}
    ```
    
    keep the context in mind when summerizing the text.
    Please answer in German.
    """

    statement_chain_prompt = PromptTemplate(template=statement_chain_prompt_template, input_variables=["CONTEXT", "TEXT"])
    statement_chain = statement_chain_prompt | ChatOpenAI(temperature=0, model=GPT_MODEL) | StrOutputParser()

    summerization_statements = statement_chain.invoke({"CONTEXT": "", "TEXT": [f"<statement_begin>{statement}<statement_end>" for statement in summeraized_statements]})

    #debug_output["statement_summerization"] = summerization_statements
    return summerization_statements

def gen_sum_teaser(story_no):
    possible_candidates_teaser: list = []
    possible_candidates_teaser.append(_get_teaser_for_story(story_no))

    teaser_chain_prompt_template = """
    You are an expert in summerizing text. You will receice a Text and have to summerize it less than 5 sentences.

    ```
    TEXT: {TEXT}
    ```

    Please answer in German.
    """

    teaser_chain_prompt = PromptTemplate(template=teaser_chain_prompt_template, input_variables=["TEXT"])
    teaser_chain = teaser_chain_prompt | ChatOpenAI(temperature=0, model=GPT_MODEL) | StrOutputParser()

    summerization_teaser = teaser_chain.invoke({"TEXT": possible_candidates_teaser[0]})

    #debug_output["teaser_summerization"] = summerization_teaser
    return summerization_teaser

def gen_radio_show_transcript(teaser_sum, statement_sums_sum):
    radio_chain_prompt_template = """
    You are an Author for a Radio Show. You will receive a Text and have to create a transcript for a radio show.
    The Transcript cannot exceed 4095 characters. 

    You will receive two types of content. The first is a teaser, which should be used to introduce the topic. The second is a
    a summerized collection of expert statement.

    Your Input is formated always as follows:
    ```
    TEASER: {TEASER}
    STATEMENTS: {STATEMENTS}
    ```
    The Output should be ready to be processed by a text-to-speech engine.

    Please answer in German.
    """

    radioc_chain_prompt = PromptTemplate(template=radio_chain_prompt_template, input_variables=["TEASER", "STATEMENTS"])
    radio_chain = radioc_chain_prompt | ChatOpenAI(temperature=0, model=GPT_MODEL) | StrOutputParser()

    radio_transcript = radio_chain.invoke({"TEASER": teaser_sum, "STATEMENTS": statement_sums_sum})

    #debug_output["radio_transcript"] = radio_transcript

    #with open(os.path.join(DATA_LOCATION, "debug_output.json"), "w") as f:
    #    json.dump(debug_output, f, indent=4)


    return radio_transcript

def gen_radio_show_audio(transcript):
    client = OpenAI()
    chunk_size = 3000

    mp3_path = "/workspace/results/radio_report_example"

    for i in range(0, len(transcript), chunk_size):
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=transcript[i:i+chunk_size]
        )

        response.stream_to_file(f"{mp3_path}_{i}.mp3")

    mp3_paths = glob.glob("results/radio_report_example_0.mp3")

    sound1 = AudioSegment.from_mp3("/home/dachman/Desktop/Test/walker.mp3")
