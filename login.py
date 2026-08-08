import streamlit as st

# Login page 
def login_page():
    st.title("Login")

    # Prompt for credentials
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state["valid_username"] = username
            st.session_state["valid_password"] = password
            st.session_state["authenticated"] = True
            st.success("Login successful! Go to the Chatbot page.")
        else:
            st.error("Please enter both username and password.")

# Run login page
login_page()