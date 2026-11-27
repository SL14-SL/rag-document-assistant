"""Shared styling for the Streamlit interface."""

import streamlit as st


def apply_ui_theme() -> None:
    """Apply consistent typography and document assistant styling."""
    st.markdown(
        """
        <style>
        /* Application subtitle */
        [data-testid="stMarkdownContainer"] p.app-subtitle {
            font-size: 1.15rem;
            line-height: 1.6;
            font-weight: 400;
            color: #526F64;
            margin-bottom: 0.75rem;
        }
        /* Caption text */
        [data-testid="stCaptionContainer"] {
            font-size: 1rem;
            line-height: 1.6;
        }

        [data-testid="stCaptionContainer"] p {
            font-size: 1rem;
            line-height: 1.6;
        }

        /* Body text and Markdown */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stText"] {
            font-size: 1.05rem;
            line-height: 1.65;
        }

        /* Widget labels */
        label,
        .stSelectbox label,
        .stTextInput label,
        .stTextArea label {
            font-size: 1rem !important;
        }

        /* Question input */
        [data-testid="stTextArea"] textarea {
            font-size: 1.05rem !important;
            line-height: 1.6 !important;
        }

        [data-testid="stTextArea"] textarea::placeholder {
            font-size: 1.05rem !important;
        }

        /* Expander headers */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary p {
            font-size: 1rem;
        }

        /* Sidebar text and captions */
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            font-size: 1rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )