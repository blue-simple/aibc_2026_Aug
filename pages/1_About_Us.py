# 1_About_Us.py
# A detailed page outlining the project scope, objectives, data sources, and features.
# Team Project Information - Autism Support Chatbot

import streamlit as st

# Project information dictionary
project = {
    "name": "Autism Support Chatbot",
    "scope": [
        "Create a chatbot for autistic patients with 24/7 conversational support",
        "Build a RAG bot to answer questions during conversations",
        "Create an agent to scrape information from various sources",
        "Assist with validation and deploy using Streamlit"
    ],
    "objectives": [
        "Provide conversational support for autistic patients",
        "Guide non-autistic people on how to care for people with autism"
    ],
    "data_sources": {
        "SG Enable": "https://www.enablingguide.sg/disability-info/autism",
        "Autism Research Institute": "https://autism.org/",
        "National Institute of Mental Health": "https://www.nimh.nih.gov/"
    },
    "features": [
        "24/7 AI chatbot for mental support through online conversations",
        "User-friendly interface customised for autistic patients"
    ]
}

# Streamlit page content
st.title("ℹ️ About Us")
st.subheader(f"Project: {project['name']}")

st.markdown("### 📌 Project Scope")
for i, scope in enumerate(project["scope"], 1):
    st.write(f"{i}. {scope}")

st.markdown("### 🎯 Objectives")
for i, obj in enumerate(project["objectives"], 1):
    st.write(f"{i}. {obj}")

st.markdown("### 📚 Data Sources for Fact Check")
for source, url in project["data_sources"].items():
    st.write(f"- [{source}]({url})")

st.markdown("### ⚙️ Features")
for i, feature in enumerate(project["features"], 1):
    st.write(f"{i}. {feature}")
