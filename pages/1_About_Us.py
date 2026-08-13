# 1_About_Us.py
# A detailed page outlining the project scope, objectives, data sources, and features.
# Team Project Information - Autism Support Chatbot

import streamlit as st
from pathlib import Path

def load_trusted_sources(file_path="data/trusted_sources.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    except FileNotFoundError:
        return []

TRUSTED_SOURCE_URLS = load_trusted_sources()

# Project information dictionary
project = {
    "name": "Autism Support Chatbot",
    "scope": [
        "Create a chatbot for autistic patients/caregivers with 24/7 conversational support",
        "Build a RAG bot to answer questions during conversations",
        "Create an agent to scrape information from various sources",
        "Assist with validation and deploy using Streamlit"
    ],
    "objectives": [
        "Provide conversational support for autistic patients/caregivers",
        "Guide caregivers on how to care for people with autism"
    ],
    "data_sources": {
        "SG Enable": TRUSTED_SOURCE_URLS[0],
        "Autism Research Institute": TRUSTED_SOURCE_URLS[1],
        "National Institute of Mental Health": TRUSTED_SOURCE_URLS[2]
    },
    "features": [
        "24/7 AI chatbot for mental support through online conversations",
        "User-friendly interface customised for autistic patients"
    ],
    "members": [
        "Chua Wei Shan",
        "Chua Yee May",        
        "Sng Boon Wei, Raymond"
    ],
    "real_world_problem": {
        "problem_statement": (
            "There is a significant gap between untrained caregivers — such as parents, "
            "family members, and volunteers — who are suddenly thrust into caring for autistic "
            "individuals, and the trained professionals who have the knowledge and skills to do "
            "so effectively. This gap leads to caregiver burnout, inconsistent care, and poorer "
            "outcomes for autistic individuals."
        ),
        "how_it_helps": [
            "Provides untrained caregivers with immediate, accessible guidance on managing "
            "day-to-day challenges with autistic individuals",
            "Offers evidence-based responses drawn from trusted sources, bridging the knowledge "
            "gap when a professional is not available",
            "Reduces caregiver anxiety by giving them a reliable first point of contact at any "
            "hour of the day",
            "Helps caregivers recognise early warning signs and understand behavioural triggers, "
            "empowering them to respond more effectively"
        ],
        "disclaimer": (
            "This chatbot is a band-aid solution, not a cure. Structured, professional training "
            "for caregivers remains essential and cannot be replaced by technology alone. "
            "What this tool can do is provide immediate relief to the caregivers who are "
            "overwhelmed today — while the longer-term work of building proper support systems "
            "and training pipelines continues. Technology is not the answer, but it can help "
            "carry the load until better solutions are in place."
        )
    }
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

st.markdown("### 👥 Project Members")
for i, member in enumerate(project["members"], 1):
    st.write(f"{i}. {member}")

st.markdown("### 🌍 How It Solves a Real-World Problem")
rw = project["real_world_problem"]

st.markdown("**The Problem**")
st.write(rw["problem_statement"])

st.markdown("**How This Chatbot Helps**")
for i, point in enumerate(rw["how_it_helps"], 1):
    st.write(f"{i}. {point}")