import streamlit as st
import os
import sys

# Safely import packages
st.title("Galaxy")
st.caption("An intelligent researcher. Galaxy helps in your research and can also assist you in writing research papers.")

# Create requirements.txt in the sidebar
with st.sidebar:
    st.header("Setup Guide")
    st.subheader("Required packages")
    st.code("""
    streamlit==1.30.0
    langchain==0.1.0
    langchain-google-genai==0.0.6
    langchain-community==0.0.13
    google-generativeai==0.3.2
    faiss-cpu==1.7.4
    wikipedia==1.4.0
    python-dotenv==1.0.0
    """, language="text")
    
    st.markdown("If you're experiencing import errors, make sure your requirements.txt file includes all the packages above.")

# Try imports one by one to provide better error messages
try:
    # No longer using dotenv
    # Simply tell users they can provide their API key directly
    pass
except ImportError:
    st.warning("python-dotenv package not found. This is fine - you can provide your API key directly.")

# Try other imports
missing_packages = []

try:
    import google.generativeai as genai
except ImportError:
    missing_packages.append("google-generativeai")

try:
    from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError:
    missing_packages.append("langchain-google-genai")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
except ImportError:
    missing_packages.append("langchain-community")

try:
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
except ImportError:
    missing_packages.append("langchain")

# If we're missing any crucial packages, show error and stop
if missing_packages:
    st.error(f"Missing required packages: {', '.join(missing_packages)}")
    st.error("Please make sure all required packages are installed in your Streamlit environment.")
    
    with st.expander("How to fix"):
        st.markdown("""
        ### Fixing Package Issues
        
        1. Make sure you have a `requirements.txt` file in your repository with all the required packages.
        2. If deploying on Streamlit Cloud, check the logs for specific error messages.
        3. Try installing a specific version of the problematic package.
        """)
        
    st.stop()

# API Key Management
with st.sidebar:
    st.header("API Configuration")
    
    gemini_api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    
    if gemini_api_key:
        os.environ["GOOGLE_GEMINI_API_KEY"] = gemini_api_key
        try:
            genai.configure(api_key=gemini_api_key)
            llm = GoogleGenerativeAI(model="gemini-pro", verbose=True)
            st.success("Google Gemini API key configured!")
        except Exception as e:
            st.error(f"Error configuring API key: {str(e)}")
            llm = None
    else:
        st.error("Please enter a valid API key to use Galaxy")
        llm = None

# Initialize other tools only if we have a valid LLM
if llm:
    try:
        wiki = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=300)
        arxiv = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=400)
        tool1 = ArxivQueryRun(api_wrapper=arxiv)
        tool = WikipediaQueryRun(api_wrapper=wiki)
        
        # Function to process documents
        def process_docs(docs):
            text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split = text_split.split_documents(docs)
            embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector = FAISS.from_documents(split, embed)
            return vector

        def process_arxiv_docs(arxiv_docs, question):
            try:
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
            except Exception as e:
                return f"Error processing arXiv data: {str(e)}"

        def process_wiki_docs(wiki_docs, question):
            try:
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
            except Exception as e:
                return f"Error processing Wikipedia data: {str(e)}"

        def ans_question(question):
            with st.status("Researching..."):
                st.write("Searching arXiv and Wikipedia...")
                try:
                    arxiv_answer = None
                    wiki_answer = None

                    st.write("Querying arXiv...")
                    arxiv_docs = arxiv.run(f"{question}")
                    if arxiv_docs:
                        st.write("Processing arXiv results...")
                        arxiv_answer = process_arxiv_docs(arxiv_docs, question)

                    st.write("Querying Wikipedia...")
                    wiki_docs = tool.run(f"{question}")
                    if wiki_docs:
                        st.write("Processing Wikipedia results...")
                        wiki_answer = process_wiki_docs(wiki_docs, question)

                    st.write("Compiling answer...")
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
            with st.status("Writing research paper..."):
                try:
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
                except Exception as e:
                    return f"Error generating research paper: {str(e)}"

        def summary(id):
            with st.status(f"Fetching and summarizing arXiv paper {id}..."):
                try:
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
                except Exception as e:
                    return f"Error generating summary: {str(e)}"

        # User Input
        user_input = st.chat_input("ask to Galaxy")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                if "write research paper" in user_input.lower():
                    outline = writer(user_input)
                    st.write(outline)
                elif "give summary" in user_input.lower() and ":" in user_input:
                    # Extract the arXiv ID
                    paper_id = user_input.split(":", 1)[1].strip()
                    sum_result = summary(paper_id)
                    st.write(sum_result)
                else:
                    response = ans_question(user_input)
                    st.write(response)
    except Exception as e:
        st.error(f"Error initializing tools: {str(e)}")
        st.info("This might be due to missing packages or API configuration issues.")
else:
    st.warning("Please configure an API key in the sidebar to use Galaxy")

