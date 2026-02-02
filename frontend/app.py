# Data-Crawler Streamlit Frontend
# State-of-the-Art UI with chat continuity and hybrid retrieval

import streamlit as st
import httpx
from pathlib import Path
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
API_V1 = f"{API_BASE_URL}/api/v1"

# Page config
st.set_page_config(
    page_title="Data Crawler",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .ai-message {
        background: white;
        border: 1px solid #e0e0e0;
        padding: 0.75rem 1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .source-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        margin: 0.2rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .status-completed { background: #4caf50; }
    .status-processing { background: #ff9800; }
    .status-failed { background: #f44336; }
    .status-queued { background: #9e9e9e; }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Session state
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def api_request(method: str, endpoint: str, data=None, files=None, form_data=None):
    """Make API request."""
    url = f"{API_V1}{endpoint}"
    headers = get_headers()
    
    try:
        with httpx.Client(timeout=120.0) as client:
            if method == "GET":
                response = client.get(url, headers=headers, params=data)
            elif method == "POST":
                if files:
                    response = client.post(url, headers=headers, files=files)
                elif form_data:
                    response = client.post(url, headers=headers, data=form_data)
                else:
                    response = client.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = client.delete(url, headers=headers)
            else:
                return None, f"Unsupported method: {method}"
            
            if response.status_code in [200, 201, 202, 204]:
                if response.status_code == 204:
                    return {}, None
                return response.json(), None
            elif response.status_code == 401:
                st.session_state.token = None
                return None, "Session expired. Please login again."
            else:
                try:
                    error = response.json().get("detail", response.text)
                except:
                    error = response.text
                return None, error
                
    except httpx.ConnectError:
        return None, "Cannot connect to API server"
    except Exception as e:
        return None, str(e)


# ============================================
# AUTH PAGE
# ============================================
def show_auth_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">📚 Data Crawler</h1>', unsafe_allow_html=True)
        st.markdown("### State-of-the-Art Document Intelligence")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Email", placeholder="user@example.com")
                password = st.text_input("Password", type="password")
                
                if st.form_submit_button("Login", use_container_width=True):
                    if username and password:
                        with st.spinner("Logging in..."):
                            data, error = api_request("POST", "/auth/login", form_data={
                                "username": username,
                                "password": password
                            })
                            if error:
                                st.error(f"Login failed: {error}")
                            else:
                                st.session_state.token = data["access_token"]
                                st.session_state.username = username
                                st.success("Welcome back!")
                                st.rerun()
        
        with tab2:
            with st.form("register_form"):
                reg_username = st.text_input("Email", placeholder="user@example.com", key="reg_user")
                reg_password = st.text_input("Password (min 8 chars)", type="password", key="reg_pass")
                reg_confirm = st.text_input("Confirm Password", type="password")
                
                if st.form_submit_button("Create Account", use_container_width=True):
                    if reg_password != reg_confirm:
                        st.error("Passwords don't match")
                    elif len(reg_password) < 8:
                        st.error("Password must be at least 8 characters")
                    elif reg_username and reg_password:
                        with st.spinner("Creating account..."):
                            data, error = api_request("POST", "/auth/signup", data={
                                "username": reg_username,
                                "password": reg_password
                            })
                            if error:
                                st.error(f"Registration failed: {error}")
                            else:
                                st.success("Account created! Please login.")


# ============================================
# SIDEBAR
# ============================================
def show_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📄 Documents", "🔍 Search", "📊 Dashboard"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Show previous chats
        st.markdown("### 💬 Recent Chats")
        chats_data, _ = api_request("GET", "/query/chats")
        if chats_data and chats_data.get("chats"):
            for chat in chats_data["chats"][:5]:
                chat_name = chat.get("name", "Untitled")[:25]
                if st.button(f"📝 {chat_name}", key=f"chat_{chat['id']}", use_container_width=True):
                    load_chat(chat["id"])
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        return page


def load_chat(chat_id: str):
    """Load a previous chat."""
    data, error = api_request("GET", f"/query/chats/{chat_id}")
    if not error and data:
        st.session_state.current_chat_id = chat_id
        st.session_state.messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in data.get("history", [])
        ]
        st.rerun()


# ============================================
# CHAT PAGE
# ============================================
def show_chat_page():
    st.markdown("## 💬 Chat with Your Documents")
    
    # Settings expander
    with st.expander("⚙️ Retrieval Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_hyde = st.checkbox("Use HyDE", value=True, help="Query expansion for better retrieval")
        with col2:
            use_bm25 = st.checkbox("Use BM25", value=True, help="Lexical search for keyword matching")
        with col3:
            top_k = st.slider("Context chunks", 3, 10, 5)
    
    # Display messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""<div class="user-message">{msg["content"]}</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="ai-message">{msg["content"]}</div>""", unsafe_allow_html=True)
                # Show sources if available
                if "sources" in msg:
                    sources_html = " ".join([
                        f'<span class="source-badge">{s["source"]}: {s["score"]:.2f}</span>'
                        for s in msg.get("sources", [])[:3]
                    ])
                    st.markdown(sources_html, unsafe_allow_html=True)
    
    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask about your documents...",
            height=100,
            placeholder="What is...? How does...? Explain...",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([5, 1])
        with col2:
            submitted = st.form_submit_button("Send 📤", use_container_width=True)
    
    if submitted and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response
        with st.spinner("🤔 Thinking..."):
            data, error = api_request("POST", "/query/", data={
                "prompt": user_input,
                "chat_id": st.session_state.current_chat_id,
                "use_hyde": use_hyde,
                "use_bm25": use_bm25,
                "top_k": top_k
            })
            
            if error:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {error}"
                })
            else:
                st.session_state.current_chat_id = data.get("chat_id")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data.get("ai_response", "No response"),
                    "sources": data.get("sources", [])
                })
        
        st.rerun()


# ============================================
# DOCUMENTS PAGE
# ============================================
def show_documents_page():
    st.markdown("## 📄 Document Library")
    
    # Upload section
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "docx", "txt", "md", "pptx", "html"],
            help="Supported: PDF, DOCX, TXT, Markdown, PPTX, HTML"
        )
    
    with col2:
        st.write("")
        st.write("")
        if uploaded_file and st.button("📤 Upload & Process", use_container_width=True):
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data, error = api_request("POST", "/documents/upload", files=files)
                
                if error:
                    st.error(f"Upload failed: {error}")
                else:
                    st.success(f"✅ Uploaded! Processing started...")
                    st.info(f"Document ID: {data.get('id')}")
                    time.sleep(1)
                    st.rerun()
    
    st.divider()
    
    # Documents list
    st.markdown("### Your Documents")
    
    data, error = api_request("GET", "/documents/")
    
    if error:
        st.warning(f"Could not load documents: {error}")
    elif not data:
        st.info("📭 No documents yet. Upload your first document above!")
    else:
        for doc in data:
            status = doc.get("status", "unknown")
            status_class = f"status-{status.lower()}"
            status_icon = {"completed": "✅", "processing": "⏳", "failed": "❌", "queued": "⏸️"}.get(status.lower(), "❓")
            
            with st.container():
                col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**📄 {doc.get('filename', 'Unknown')}**")
                
                with col2:
                    st.markdown(f"{status_icon} {status.capitalize()}")
                
                with col3:
                    created = doc.get("created_at", "")
                    if created:
                        try:
                            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                            st.caption(dt.strftime("%b %d, %H:%M"))
                        except:
                            st.caption(created[:10])
                
                with col4:
                    if st.button("🗑️", key=f"del_{doc.get('id')}", help="Delete document"):
                        _, del_error = api_request("DELETE", f"/documents/{doc.get('id')}")
                        if del_error:
                            st.error(del_error)
                        else:
                            st.rerun()
                
                st.divider()


# ============================================
# SEARCH PAGE
# ============================================
def show_search_page():
    st.markdown("## 🔍 Semantic Search")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Search your documents...", placeholder="Enter search query")
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=50, value=10)
    
    use_hyde = st.checkbox("Use HyDE (query expansion)", value=False)
    
    if st.button("🔍 Search", use_container_width=True) and query:
        with st.spinner("Searching..."):
            data, error = api_request("GET", "/query/search", data={
                "query": query,
                "top_k": top_k,
                "use_hyde": use_hyde
            })
            
            if error:
                st.error(error)
            elif data:
                results = data.get("results", [])
                
                if not results:
                    st.info("No results found. Try different keywords or upload more documents.")
                else:
                    st.success(f"Found {len(results)} results")
                    
                    for i, r in enumerate(results, 1):
                        score = r.get("score", 0)
                        
                        with st.expander(f"#{i} • Score: {score:.4f} • via {r.get('source', 'vector')}", expanded=i<=3):
                            st.caption(f"Document: {r.get('doc_id', 'unknown')[:16]}...")
                            st.markdown(r.get("chunk_text", "No content"))


# ============================================
# DASHBOARD PAGE
# ============================================
def show_dashboard_page():
    st.markdown("## 📊 System Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    # Services status
    with col1:
        st.markdown("### 🔧 Services")
        
        services = [
            ("Orchestrator", "http://localhost:8000"),
            ("LLM", "http://localhost:8001"),
            ("Embedding", "http://localhost:8002"),
            ("Docling", "http://localhost:8004"),
        ]
        
        for name, url in services:
            try:
                with httpx.Client(timeout=3.0) as client:
                    r = client.get(f"{url}/health" if "8000" not in url else url)
                    if r.status_code == 200:
                        st.success(f"✅ {name}")
                    else:
                        st.warning(f"⚠️ {name}")
            except:
                st.error(f"❌ {name}")
    
    # Vector stats
    with col2:
        st.markdown("### 📈 Vector Database")
        data, _ = api_request("GET", "/query/stats")
        if data:
            st.metric("Total Vectors", data.get("num_entities", 0))
            st.metric("Index Type", data.get("index_type", "HNSW"))
            st.caption(f"Mode: {data.get('mode', 'lite')}")
    
    # Documents
    with col3:
        st.markdown("### 📚 Documents")
        data, _ = api_request("GET", "/documents/")
        if data:
            total = len(data)
            completed = sum(1 for d in data if d.get("status") == "COMPLETED")
            processing = sum(1 for d in data if d.get("status") == "PROCESSING")
            
            st.metric("Total", total)
            st.metric("Completed", completed)
            if processing:
                st.metric("Processing", processing)


# ============================================
# MAIN
# ============================================
def main():
    if not st.session_state.token:
        show_auth_page()
    else:
        page = show_sidebar()
        
        if page == "💬 Chat":
            show_chat_page()
        elif page == "📄 Documents":
            show_documents_page()
        elif page == "🔍 Search":
            show_search_page()
        elif page == "📊 Dashboard":
            show_dashboard_page()


if __name__ == "__main__":
    main()
