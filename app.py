import streamlit as st
from agent_chatbot import ControllableRAGAgent

st.title("🔬 Gasoline Fuel Injection Agent Chatbot (Groq-powered RAG)")
st.markdown("Ask technical engine/fuel injection questions! Uses a controllable, stepwise agent for grounded, non-hallucinated answers.")

if "agent" not in st.session_state:
    st.session_state.agent = ControllableRAGAgent()
    st.session_state.agent.load_documents('./documents/')  # Adapt to your docs folder

query = st.text_input("Your question about fuel injection theory or practice:")
if st.button("Ask Agent") and query:
    with st.spinner("Thinking..."):
        resp = st.session_state.agent.query(query)
        st.markdown(f"**Answer:**\n{resp['answer']}")
        with st.expander("Stepwise reasoning plan:"):
            for i, step in enumerate(resp['plan']):
                st.markdown(f"- {step}")
