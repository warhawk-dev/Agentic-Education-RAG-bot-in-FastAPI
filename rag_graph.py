import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ── Models ────────────────────────────────────────────────────────────────────

llm        = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

physics_db = Chroma(
    persist_directory="chroma_db/physics",
    embedding_function=embeddings,
)

chemistry_db = Chroma(
    persist_directory="chroma_db/chemistry",
    embedding_function=embeddings,
)

biology_db = Chroma(
    persist_directory="chroma_db/biology",
    embedding_function=embeddings,
)

SUBJECT_STORES = {
    "physics": physics_db,
    "chemistry": chemistry_db,
    "biology": biology_db,
}

# ── PDF indexing helper ───────────────────────────────────────────────────────

def index_pdf(pdf_path: str, subject: str) -> int:
    """
    Reads a PDF and indexes it into the matching ChromaDB store.
    Returns the number of chunks created.
    """
    store = SUBJECT_STORES[subject]

    # Step A: Load all pages from the PDF
    documents = PyPDFLoader(pdf_path).load()

    # Step B: Split pages into smaller overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    # Step C: Add chunks to ChromaDB
    store.add_documents(chunks)
    return len(chunks)

# ── Retriever tools ───────────────────────────────────────────────────────────

@tool
def physics_retriever(query: str) -> str:
    """
    Search the Physics textbook for relevant content.
    Use this tool for questions about:
    - Mechanics: motion, velocity, acceleration, force, friction, Newton's laws
    - Energy: kinetic energy, potential energy, work, power, conservation of energy
    - Waves: sound waves, light waves, frequency, amplitude, wavelength
    - Optics: reflection, refraction, lenses, mirrors
    - Electricity & Magnetism: current, voltage, resistance, circuits, magnetic fields
    - Modern Physics: atomic structure, radioactivity, nuclear reactions
    Do NOT use this for chemistry or biology questions.
    """
    docs = physics_db.similarity_search(query, k=4)
    formatted_chunks = []
    for d in docs:
        source = d.metadata.get("source", "Physics.pdf")
        page = d.metadata.get("page", "?")
        text = d.page_content
        formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
    return "\n\n".join(formatted_chunks)


@tool
def chemistry_retriever(query: str) -> str:
    """
    Search the Chemistry textbook for relevant content.
    Use this tool for questions about:
    - Atomic Structure: atoms, electrons, protons, neutrons, orbitals, electron configuration
    - Periodic Table: elements, groups, periods, atomic number, atomic mass
    - Chemical Bonding: ionic bonds, covalent bonds, hydrogen bonds, valence electrons
    - Reactions: chemical equations, balancing, oxidation, reduction, acids, bases
    - Organic Chemistry: carbon compounds, benzene, hydrocarbons, functional groups
    - States of Matter: solids, liquids, gases, phase changes, intermolecular forces
    Do NOT use this for physics or biology questions.
    """
    docs = chemistry_db.similarity_search(query, k=4)
    formatted_chunks = []
    for d in docs:
        source = d.metadata.get("source", "Chemistry.pdf")
        page = d.metadata.get("page", "?")
        text = d.page_content
        formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
    return "\n\n".join(formatted_chunks)


@tool
def biology_retriever(query: str) -> str:
    """
    Search the Biology textbook for relevant content.
    Use this tool for questions about:
    - Cell Biology: cell structure, cell membrane, mitochondria, nucleus, organelles
    - Genetics: DNA, RNA, genes, chromosomes, heredity, mutations, genetic disorders
    - Evolution: natural selection, adaptation, Charles Darwin, species, fossils
    - Human Body: organ systems, digestion, respiration, circulation, nervous system
    - Plants: photosynthesis, chlorophyll, plant cells, transpiration, reproduction
    - Ecology: food chains, ecosystems, biodiversity, population, environment
    Do NOT use this for physics or chemistry questions.
    """
    docs = biology_db.similarity_search(query, k=4)
    formatted_chunks = []
    for d in docs:
        source = d.metadata.get("source", "Biology.pdf")
        page = d.metadata.get("page", "?")
        text = d.page_content
        formatted_chunks.append(f"[Source: {source} | Page: {page}]\n{text}")
    return "\n\n".join(formatted_chunks)


all_tools = [physics_retriever, chemistry_retriever, biology_retriever]

# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt="You are a helpful education assistant. Use the retrieval tools to find relevant textbook content.",
)

# ── RAG State ─────────────────────────────────────────────────────────────────

class RAGState(BaseModel):
    question:       str  = Field(default="")
    retrieved_text: str  = Field(default="")
    is_relevant:    bool = Field(default=False)
    final_answer:   str  = Field(default="")
    retry_count:    int  = Field(default=0)

# ── Graph nodes ───────────────────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": state.question}]})
    retrieved_text = result["messages"][-2].content   # ToolMessage is always second to last
    return {"retrieved_text": retrieved_text}


def grade_node(state: RAGState) -> dict:
    grade_chain = (
        PromptTemplate.from_template(
            "Question: {question}\n\n"
            "Passages:\n{passages}\n\n"
            "Are these passages relevant to answer the question? Reply YES or NO only."
        )
        | llm
        | StrOutputParser()
    )
    answer = grade_chain.invoke({
        "question": state.question,
        "passages": state.retrieved_text[:1500],
    })
    return {"is_relevant": "YES" in answer.upper()}


def rephrase_node(state: RAGState) -> dict:
    rephrase_chain = (
        PromptTemplate.from_template(
            "Rewrite this question to be more specific for a textbook search.\n"
            "Return ONLY the rewritten question.\n\n"
            "Original: {question}"
        )
        | llm
        | StrOutputParser()
    )
    new_question = rephrase_chain.invoke({"question": state.question})
    return {"question": new_question, "retry_count": state.retry_count + 1}


def generate_node(state: RAGState) -> dict:
    generate_chain = (
        PromptTemplate.from_template(
            "Answer the question using the passages below.\n\n"
            "Question: {question}\n\n"
            "Passages:\n{passages}\n\n"
            "Use this format:\n"
            "Answer\n-------\n<answer>\n\n"
            "Summary\n-------\n<one sentence>\n\n"
            "Source\n------\n<PDF name>\n\n"
            "Page\n----\n<page number>"
        )
        | llm
        | StrOutputParser()
    )
    answer = generate_chain.invoke({
        "question": state.question,
        "passages": state.retrieved_text,
    })
    return {"final_answer": answer}

# ── Routing ───────────────────────────────────────────────────────────────────

def after_grade(state: RAGState) -> str:
    if state.is_relevant or state.retry_count >= 1:
        return "generate"
    else:
        return "rephrase"

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("grade",    grade_node)
graph.add_node("rephrase", rephrase_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", after_grade, {"generate": "generate", "rephrase": "rephrase"})
graph.add_edge("rephrase", "retrieve")
graph.add_edge("generate", END)

rag_app = graph.compile()