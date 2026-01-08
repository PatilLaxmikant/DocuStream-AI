import streamlit as st
import os
import tempfile
from rag_engine import RAGEngine

import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

st.set_page_config(page_title="DocuStream AI", layout="wide")

# --- Authentication ---
from pathlib import Path
script_dir = Path(__file__).parent
auth_config_path = script_dir / 'auth_config.yaml'

with open(auth_config_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render the valid login widget
# name, authentication_status, username = authenticator.login("Login", "main")
# NOTE: The API might return different values depending on version. 
# Safe check:
result = authenticator.login()
if isinstance(result, tuple):
    name, authentication_status, username = result
else:
    # Older versions or different signature
    authentication_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")


if authentication_status is False:
    st.error("Username/password is incorrect")
    st.stop()
elif authentication_status is None:
    st.warning("Please enter your username and password")
    st.stop()
    
# If we get here, the user is logged in
st.sidebar.write(f"Welcome *{name}*")
authenticator.logout("Logout", "sidebar")
st.sidebar.divider()

st.title("📚 DocuStream AI")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()

# Sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Process File"):
            with st.spinner("Indexing..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Index
                try:
                    result = st.session_state.engine.index_file(tmp_path)
                    if isinstance(result, dict):
                        if result["status"] == "success":
                            st.success(f"Successfully indexed {result['chunks']} chunks!")
                        elif result["status"] == "skipped":
                            st.warning(result["message"])
                        else:
                            st.error(f"Error: {result['message']}")
                    else:
                         st.success(f"Successfully indexed {result} chunks!")
                except Exception as e:
                    st.error(f"Error indexing: {e}")
                finally:
                    os.unlink(tmp_path)

    st.divider()
    st.markdown("### Management")
    if st.button("Clear Database", type="primary"):
        with st.spinner("Clearing database..."):
            result = st.session_state.engine.clear_database()
            if "success" in result.lower():
                st.success(result)
                st.session_state.messages = [] # Clear chat history too
                st.rerun()
            else:
                st.error(result)

    st.divider()
    st.markdown("### Database Stats")
    stats = st.session_state.engine.get_collection_stats()
    if stats.get("status") == "Error":
         st.error(f"Error: {stats.get('error')}")
    else:
        st.metric("Total Vectors", stats.get("vectors", 0))
        st.caption(f"Status: {stats.get('status')}")

    st.divider()
    st.markdown("### Configuration")
    if st.session_state.engine.qdrant_url and "localhost" not in st.session_state.engine.qdrant_url:
         st.success("✅ Connected to Qdrant Cloud")
    else:
         st.info("🏠 Using Local Qdrant")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your document..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources = []
        
        # Stream response
        try:
            stream = st.session_state.engine.chat_stream(prompt)
            for item in stream:
                if item["type"] == "sources":
                    sources = item["content"]
                elif item["type"] == "chunk":
                    full_response += item["content"]
                    response_placeholder.markdown(full_response + "▌")
                elif item["type"] == "error":
                    st.error(item["content"])
                    full_response = "Error generating response."
            
            response_placeholder.markdown(full_response)
            
            # Show Citations
            if sources:
                with st.expander("📚 Referenced Sources"):
                    for src in sources:
                        st.caption(f"**Page {src['page']}** ({src['source']})")
                        st.markdown(f"> {src['text']}")
                        st.divider()
                        
        except Exception as e:
            st.error(f"Stream error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
