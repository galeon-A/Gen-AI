import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_google_genai import GoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document

load_dotenv()

st.title("Galaxy")
st.caption("An intelligent researcher. Galaxy helps in your research and can also assist you in writing research papers.")


with st.sidebar:
    st.header("API Configuration")
    api_option = st.radio(
        "Select API Provider:",
        ("Google Gemini", "OpenAI")
    )
    
    if api_option == "Google Gemini":
        gemini_api_key = st.text_input("Enter Google Gemini API Key:", type="password", 
                                       value=os.getenv("GOOGLE_GEMINI_API_KEY", ""))
        if gemini_api_key:
            os.environ["GOOGLE_GEMINI_API_KEY"] = gemini_api_key
            genai.configure(api_key=gemini_api_key)
            llm = GoogleGenerativeAI(model="gemini-pro", verbose=True)
            st.success("Google Gemini API key configured!")
        else:
            st.error("Please enter a valid API key to use Galaxy")
            llm = None
    else:  
        openai_api_key = st.text_input("Enter OpenAI API Key:", type="password",
                                      value=os.getenv("OPENAI_API_KEY", ""))
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            llm = ChatOpenAI(temperature=0.1)
            st.success("OpenAI API key configured!")
        else:
            st.error("Please enter a valid API key to use Galaxy")
            llm = None


if llm:
    wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=300)
    arxiv = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=400)
    tool1 = ArxivQueryRun(api_wrapper=arxiv)
    tool = WikipediaQueryRun(api_wrapper=wiki)

 
    def process_docs(docs):
        text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split = text_split.split_documents(docs)
        
    
        if api_option == "Google Gemini":
            embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        else:  
            from langchain.embeddings import OpenAIEmbeddings
            embed = OpenAIEmbeddings()
            
        vector = FAISS.from_documents(split, embed)
        return vector

    def process_arxiv_docs(arxiv_docs, question):
        arxiv_vector = process_docs([Document(page_content=arxiv_docs)])
        relevant_arxiv = arxiv_vector.similarity_search(question)

        arxiv_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""using the following context from arXiv, 
            answer the following question in details
            context:{context}
            question:{question}
            answer:
            """
        )
        arxiv_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=arxiv_prompt)
        answer = arxiv_chain({"input_documents": relevant_arxiv, "question": question}, return_only_outputs=True)
        return answer["output_text"]

    def process_wiki_docs(wiki_docs, question):
        wiki_vector = process_docs([Document(page_content=wiki_docs)])
        relevant_wiki = wiki_vector.similarity_search(question)

        wiki_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""using the following context from Wikipedia, 
            answer the following question in details
            context:{context}
            question:{question}
            answer:
            """
        )
        wiki_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=wiki_prompt)
        answer = wiki_chain({"input_documents": relevant_wiki, "question": question}, return_only_outputs=True)
        return answer["output_text"]

    def ans_question(question):
        try:
            arxiv_answer = None
            wiki_answer = None

            arxiv_docs = arxiv.run(f"{question}")
            if arxiv_docs:
                arxiv_answer = process_arxiv_docs(arxiv_docs, question)

            wiki_docs = tool.run(f"{question}")
            if wiki_docs:
                wiki_answer = process_wiki_docs(wiki_docs, question)

            if arxiv_answer and wiki_answer:
                return f" By ArXiv :\n{arxiv_answer}\n\n By Wikipedia :\n{wiki_answer}"
            elif arxiv_answer:
                return f"Based on ArXiv:\n{arxiv_answer}"
            elif wiki_answer:
                return f"Based on Wikipedia:\n{wiki_answer}"
            else:
                return "I couldn't find any relevant information to answer your question."
        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}. Please try again."

    def writer(topic):
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
    You are a expert researcher and skilled research paper writer.
    User will give you any {topic} to write research paper on.
    You have to write a detailed paper with all the information.
    1. Introduction
    2. Background/Literature Review
    3. Methodology (if applicable)
    4. Main body (with several key points or arguments)
    5. Discussion
    6. Conclusion

    Be specific to the topic of {topic} and provide a detailed structure.
    """
        )

        new_prompt = prompt.format(topic=topic)
        response = llm.invoke(new_prompt)
        return response

    def summary(id):
        paper = arxiv.run(f"id:{id}")
        summary_prompt = PromptTemplate(
            input_variables=["paper"],
            template="""
    Summarize the following arXiv paper in detail
    the user will give the {paper} and you have to summarize it
    """
        )

        summary_chain = load_qa_chain(llm, chain_type="stuff")
        paper_summary = summary_chain(
            {"input_documents": [Document(page_content=paper)], "question": summary_prompt.format(paper=paper)},
            return_only_outputs=True
        )
        return paper_summary['output_text']

 
    user_input = st.chat_input("ask to Galaxy")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            if "write research paper" in user_input.lower():
                outline = writer(user_input)
                st.write(outline)
            elif "give summary" in user_input.lower():
                sum = summary(user_input)
                st.write(sum)
            else:
                response = ans_question(user_input)
                st.write(response)
else:
    st.warning("Please configure an API key in the sidebar to use Galaxy")
