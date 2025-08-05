import streamlit as st
import streamlit.components.v1 as components

raw_html = """
<div style="background-color: #dff0d8; padding: 20px; border-radius: 10px;">
  <h4>✅ Hello from HTML</h4>
  <p>This is a fully rendered HTML component.</p>
</div>
"""

st.write("Normal Streamlit Text")
components.html(raw_html, height=200)
