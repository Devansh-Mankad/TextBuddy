import json
import streamlit as st
from pipeline import process
from spell_checking.spell_checker import SpellChecker

st.set_page_config(page_title="TextBuddy - Smart Reading & Writing Assistant",layout="wide")
checker=SpellChecker()

def render_tree(node,level=0):
    if isinstance(node,str):
        return "&nbsp;"*(level*4)+f"<span>{node}</span>"
    label=node.get("label","?")
    html=f"<div style='margin-left:{level*22}px'><b>{label}</b></div>"
    for child in node.get("children",[]):
        html+=render_tree(child,level+1)
    return html

st.title("TextBuddy - Smart Reading & Writing Assistant")
st.caption("NLP pipeline + bonus autocomplete-as-you-type demo")
text=st.text_area("Write or paste text", "The stud", height=180)

last=""
m=__import__("re").search(r"([A-Za-z]+)$",text)
if m: last=m.group(1)
if last:
    suggestions=[w for w,d in checker.candidates(last,limit=5)]
    st.subheader("Autocomplete suggestions")
    if suggestions:
        st.write(" • ".join(suggestions))
    else:
        st.write("No suggestions")

if st.button("Run full NLP pipeline"):
    result=process(text)
    st.subheader("Corrected text")
    st.write(result["corrected_text"])
    st.subheader("Summary")
    st.json(result["summary"])
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Spelling diff")
        st.json(result["spelling_changes"])
        st.subheader("Semantic frames")
        st.json(result["semantic_frames"])
    with c2:
        st.subheader("Discourse / coreference / pragmatics")
        st.json({
            "discourse_relations":result["discourse_relations"],
            "coref_chains":result["coref_chains"],
            "pragmatic_inferences":result["pragmatic_inferences"]
        })
    st.subheader("CFG parse trees")
    for i,tree in enumerate(result["parse_trees"],1):
        st.markdown(f"**Sentence {i}**")
        st.markdown(render_tree(tree),unsafe_allow_html=True)
    st.subheader("Complete JSON")
    st.code(json.dumps(result,indent=2,default=str),language="json")
