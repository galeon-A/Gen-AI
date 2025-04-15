import streamlit as st
import os

# Set up the basic UI
st.title("Galaxy Research Assistant")
st.caption("An intelligent researcher that helps with your research and writing.")

# Create diagnostic section at the top
st.warning("⚠️ This is running in diagnostic mode to help identify import issues.")

# Package import diagnostics
packages_to_check = [
    "google.generativeai",
    "langchain",
    "langchain.chains.question_answering",
    "langchain.prompts",
    "langchain.text_splitter",
    "langchain.docstore.document",
    "langchain_google_genai",
    "langchain_community.vectorstores",
    "langchain_community.utilities",
    "langchain_community.tools",
    "faiss"
]

# Check each package individually
st.write("### Package Import Diagnostics")
import_results = {}

for package in packages_to_check:
    try:
        exec(f"import {package}")
        import_results[package] = "✅ Success"
    except ImportError as e:
        import_results[package] = f"❌ Failed: {str(e)}"
    except Exception as e:
        import_results[package] = f"⚠️ Error: {str(e)}"

# Display results in an expander
with st.expander("View Import Results"):
    for package, result in import_results.items():
        st.write(f"**{package}**: {result}")

# Check if key packages are available
can_import_gemini = "✅" in import_results.get("google.generativeai", "")
can_import_langchain = "✅" in import_results.get("langchain", "")
can_import_langchain_google = "✅" in import_results.get("langchain_google_genai", "")

# Simple API key input
with st.sidebar:
    st.header("API Configuration")
    gemini_api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    
    if gemini_api_key:
        os.environ["GOOGLE_GEMINI_API_KEY"] = gemini_api_key
        if can_import_gemini and can_import_langchain_google:
            st.success("API key saved. Run a simple test below.")
        else:
            st.error("API key saved, but required packages are missing.")
    else:
        st.info("Please enter your Google Gemini API key.")

# Simple test function if packages are available
if gemini_api_key and can_import_gemini and can_import_langchain_google and can_import_langchain:
    try:
        import google.generativeai as genai
        from langchain_google_genai import GoogleGenerativeAI
        
        st.write("### Test Basic Functionality")
        test_prompt = st.text_input("Enter a test prompt:", value="Explain what research is in one paragraph.")
        
        if st.button("Run Test"):
            with st.spinner("Running test..."):
                try:
                    genai.configure(api_key=gemini_api_key)
                    llm = GoogleGenerativeAI(model="gemini-pro")
                    response = llm.invoke(test_prompt)
                    st.success("Test successful!")
                    st.write("### Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Test failed: {str(e)}")
    except Exception as e:
        st.error(f"Failed to initialize test function: {str(e)}")
else:
    missing = []
    if not can_import_gemini:
        missing.append("google-generativeai")
    if not can_import_langchain:
        missing.append("langchain")
    if not can_import_langchain_google:
        missing.append("langchain-google-genai")
    
    if missing:
        st.error(f"Missing required packages: {', '.join(missing)}")
    
    if not gemini_api_key:
        st.info("Enter an API key in the sidebar to continue.")

# Provide requirements.txt information
with st.expander("Requirements Information"):
    st.markdown("""
    ### Required Packages
    
    Copy this content to your `requirements.txt` file:
    ```
    streamlit==1.31.0
    google-generativeai==0.3.2
    langchain==0.1.0
    langchain-google-genai==0.0.6
    langchain-community==0.0.13
    faiss-cpu==1.7.4
    ```
    
    ### Installation
    
    You can install these packages locally with:
    ```
    pip install -r requirements.txt
    ```
    """)

# Troubleshooting tips
with st.expander("Troubleshooting Tips"):
    st.markdown("""
    ### Common Issues
    
    1. **Import Errors**: Make sure all packages are properly installed and listed in requirements.txt
    
    2. **Package Conflicts**: Try creating a fresh virtual environment for this project
    
    3. **Streamlit Cloud Issues**: 
       - Check the logs for detailed error messages
       - Try specifying exact package versions
       - Make sure your requirements.txt file is at the root of your repository
    
    4. **API Key Issues**: Verify your API key is valid and has the necessary permissions
    
    5. **Memory Issues**: If you're getting memory errors, you might need to upgrade your Streamlit plan
    """)
