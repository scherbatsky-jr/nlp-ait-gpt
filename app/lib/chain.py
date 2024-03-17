# chain.py

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
import os
from langchain import PromptTemplate, HuggingFacePipeline

class ChatChain:
    def __init__(self):
        prompt_template = """
            "I am AIT GPT, here to answer your questions!"
            {context}
            Question: {question}
            Answer:
            """.strip()

        self.PROMPT = PromptTemplate.from_template(template=prompt_template)

        vector_path = '../../vector-store'
        db_file_name = 'nlp_stanford'
        device = "mps"

        embedding_model = HuggingFaceInstructEmbeddings(
            model_name='hkunlp/instructor-base',
            model_kwargs={"device": device}
        )

        vectordb = FAISS.load_local(
            folder_path=os.path.join(vector_path, db_file_name),
            embeddings=embedding_model,
            index_name='nlp'  # default index
        )

        self.retriever = vectordb.as_retriever()

        self.history = ChatMessageHistory()

        self.tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "lmsys/fastchat-t5-3b-v1.0",
            device_map='mps',
            load_in_8bit=False
        )

        self.pipe = pipeline(
            task="text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            model_kwargs={
                "temperature": 0,
                "repetition_penalty": 1.5
            }
        )

        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        self.question_generator = LLMChain(
            llm=self.llm,
            prompt=CONDENSE_QUESTION_PROMPT,
            verbose=True
        )

        self.query = 'Comparing both of them'
        self.chat_history = "Human:What is Machine Learning\nAI:\nHuman:What is Deep Learning\nAI:"

        self.question_generator({'chat_history' : self.chat_history, "question" : self.query})

        self.doc_chain = load_qa_chain(
            llm=self.llm,
            chain_type='stuff',
            prompt=CONDENSE_QUESTION_PROMPT,
            verbose=True
        )

        self.memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        self.chain = ConversationalRetrievalChain(
            retriever=self.retriever,
            question_generator=self.question_generator,
            combine_docs_chain=self.doc_chain,
            return_source_documents=True,
            memory=self.memory,
            verbose=True,
            get_chat_history=lambda h: h
        )

    def get_chat_chain(self):
        return self.chain
