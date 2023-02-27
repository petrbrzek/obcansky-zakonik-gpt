import streamlit as st
import os
import deepl
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import json
import re

load_dotenv()

paragraphs = json.load(open('data/data.json'))

translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))

embedding = OpenAIEmbeddings()
vectordb = Chroma(embedding_function=embedding,
                  persist_directory='db', collection_name='obcansky-zakonik')


st.write('# Ptej se občanského zákoníku 🤖')

title = st.text_input('Jaký máš dotaz?', placeholder='např. "Kdo je občan?"')

if title:
    translate_title = translator.translate_text(title, target_lang="en-us")
    query = translate_title.text
    docs = vectordb.similarity_search(query, k=2)

    chain = load_qa_chain(
        OpenAI(temperature=0), chain_type="stuff")
    output = chain({"input_documents": docs, "question": query})
    result = translator.translate_text(output['output_text'], target_lang="cs")

    st.write(result.text)

    st.write('## Zdroje')
    for doc in docs:
        par_signs = re.findall(r'§ \d+\s+\n', doc.metadata['source'])
        par_signs = [par_sing.strip() for par_sing in par_signs]
        for par_sign in par_signs:
            st.write(f'### {par_sign}')
            st.write(paragraphs[par_sign])
