import streamlit as st

def home_page():
    st.title("Welcome to the Autism Resource Chatbot")

    st.write("""
    This app helps you explore autism-related resources in Singapore.
    You can:
    - Log in with your credentials
    - Access the chatbot for queries
    - Upload documents or provide URLs for context
    """)

    st.info("Use the sidebar to navigate to **Login** or **Chatbot**.")

# Run the home page
home_page()