import streamlit as st

def home_page():
    st.title("Welcome to Autism Resource Chatbot")

    st.write("""
    This app helps you explore autism-related resources in Singapore.
    You can:
    - Log in with your credentials
    - Access the chatbot for queries
    - Upload documents or provide URLs for context
    """)

    st.info("Use the sidebar to navigate to **About Us**, **Methodology**, **Login** or **Chatbot**.")

    st.markdown("---")
    st.markdown(
        """
        <div style="background-color:#e6f2ff; padding:15px; border-radius:5px;">
        <strong>IMPORTANT NOTICE:</strong> This web application is a prototype developed for <strong>educational purposes only</strong>.  
        The information provided here is <strong>NOT intended for real-world usage</strong> and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.
        <br><br>
        <strong>Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.</strong>
        <br><br>
        Always consult with qualified professionals for accurate and personalised advice.
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the home page
home_page()