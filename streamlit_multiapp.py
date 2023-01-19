"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func, params):
        self.apps.append({
            "title": title,
            "function": func,
            "params": params
        })

    def run(self):
        st.markdown(
        """<style>
        .boxBorder1 {
            outline-offset: 20px;
            font-size:128px;
        }</style>
        """, unsafe_allow_html=True) 
        from st_btn_select import st_btn_select
        app = st_btn_select(
            self.apps,
            format_func=lambda app: '{}'.format(app['title']),
        )
        app['function'](**app['params'])