# =====================================================
# FRONTEND/APP.PY - Streamlit Frontend
# =====================================================
# Simple web interface for asking questions
# User → Question → Streamlit → API Call → Answer → Display
#
# Run: uv run streamlit run frontend/app.py

import streamlit as st
import requests

# =====================================================
# PAGE CONFIGURATION
# =====================================================
# Configure page title, icon, and layout
st.set_page_config(
    page_title="Rainbow Bazaar - Returns Assistant",
    page_icon="🛍️",
    layout="wide",
)

# Display main title
st.markdown("# 🛍️ Rainbow Bazaar Returns & Refunds Assistant")
st.markdown("Ask questions about our Returns, Refunds, and Cancellation Policy!")


# =====================================================
# SIDEBAR - HELP & EXAMPLES
# =====================================================
# Sidebar on the left side of the page
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("This AI assistant answers questions about Rainbow Bazaar's policy using our PDF document.")
    
    st.markdown("### 🎯 Example Questions")
    # Show example questions that users can ask
    examples = [
        "What is the return policy?",
        "How long do I have to return items?",
        "What items cannot be returned?",
        "What is the refund process?",
    ]
    for ex in examples:
        st.write(f"- {ex}")


# =====================================================
# BACKEND CONNECTION CHECK
# =====================================================
# Backend API URL (must be running on this URL)
API_URL = "http://localhost:8000"

# Try to connect to backend health endpoint
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    api_running = response.status_code == 200
except:
    # Connection failed - backend not running
    api_running = False

# Show status message to user
if not api_running:
    # Backend not running - show error message
    st.error("❌ Backend API is not running!")
    st.code("uv run python -m uvicorn backend.main:app --reload")
    st.stop()  # Stop execution here

# Backend is running - show success
st.success("✅ Backend API is running")


# =====================================================
# MAIN CONTENT - QUESTION INPUT
# =====================================================
st.markdown("---")

# Input field for user's question
question = st.text_input(
    "Ask your question:",
    placeholder="e.g., What is the return policy?",
)

# Button to submit question
if st.button("🔍 Search", use_container_width=True):
    # Check if question is empty
    if not question.strip():
        st.warning("Please enter a question!")
    else:
        # Show loading spinner while waiting for answer
        with st.spinner("🤔 Looking for answer..."):
            try:
                # Call backend API with the question
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=30
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    # Parse the JSON response
                    data = response.json()
                    
                    # Display the answer
                    st.markdown("### ✅ Answer")
                    st.success(data["answer"])
                else:
                    # API returned an error
                    st.error(f"Error: {response.text}")
            
            except requests.exceptions.ConnectionError:
                # Can't connect to backend
                st.error("Cannot connect to API. Make sure backend is running!")
            except Exception as e:
                # Other errors
                st.error(f"Error: {str(e)}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>🛍️ Rainbow Bazaar RAG App</div>", unsafe_allow_html=True)