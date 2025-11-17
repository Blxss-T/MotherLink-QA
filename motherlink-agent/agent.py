# from google.adk.agents.llm_agent import Agent
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# import os

# # --- Load vectorstores ---
# print("Loading vectorstores...")
# embedding_function = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")

# maternal_store = FAISS.load_local(
#     "vector_store/faiss_index",
#     embeddings=embedding_function,
#     allow_dangerous_deserialization=True
# )
# books_store = FAISS.load_local(
#     "vector_store/books_faiss_index",
#     embeddings=embedding_function,
#     allow_dangerous_deserialization=True
# )

# # Combine all vectorstores
# vectorstores = [maternal_store, books_store]

# print("Vectorstores loaded successfully!")

# # --- Prompt Template ---
# template = """You are MotherLink QA assistant. Use the context below to answer clearly in Kinyarwanda if possible.
# Context: {context}
# Question: {question}
# Answer:"""
# prompt = ChatPromptTemplate.from_template(template)

# # --- Function to query vectorstores ---
# def query_vectorstores(question, k=3):
#     all_results = []

#     for store in vectorstores:
#         # Proper retriever
#         retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})

#         # Get relevant documents - FIXED
#         docs = retriever.invoke(question)

#         # Combine contexts but limit total tokens/characters to avoid 413 errors
#         context_text = ""
#         max_chars = 800  # Reduced for shorter context (was 2000)
#         for d in docs:
#             if len(context_text) + len(d.page_content) > max_chars:
#                 break
#             context_text += d.page_content + "\n"

#         all_results.append(context_text)

#     return "\n".join(all_results)


# # --- Initialize ADK Agent ---
# root_agent = Agent(
#     model='gemini-2.5-flash',
#     name='motherlink',
#     description="Motherlink - Umufasha w'ababyeyi n'abana.",
#     instruction="""Witwa Motherlink QA. Uri umufasha ufasha ababyeyi n'abana gusubiza ibibazo bibaza ku buzima bwabo.

# ICYITONDERWA CY'INGENZI: Igisubizo cyose utanga, GIHURIZE BURI GIHE n'ababyeyi n'abana. Sobanura uko igisubizo gihurira n'ububyeyi, gutwita, cyangwa imyororokere y'abana.

# Urugero: Niba babajije kuri vitamini, sobanura uko vitamini ifasha ababyeyi batwite cyangwa abana.

# IMPORTANTE: Tanga ibisubizo bigufi cyane (maximum 160 characters) kuko bizakoreshwa kuri USSD.
# - Subiza mu nteruro 2-3 gusa
# - Koresha amagambo yoroshye
# - Ntukoreshe urutonde rurerure
# - Ibanze ku gisubizo cy'ibanze gusa kandi ugihurize n'ababyeyi/abana""",
#     tools=[]
# )


# # --- Custom query function that uses vectorstores ---
# def ask_motherlink(question):
#     """Query motherlink with RAG context from vectorstores"""
    
#     # Get context from vectorstores
#     context = query_vectorstores(question)
    
#     # Prompt in Kinyarwanda
#     prompt_text = f"""
#     Uri umufasha wa MotherLink QA. Koresha amakuru ari hasi kugirango usubize neza mu Kinyarwanda gisesuye.
    
#     ICYITONDERWA CY'INGENZI: Igisubizo cyose utanga, GIHURIZE BURI GIHE n'ababyeyi n'abana. 
#     - Sobanura uko igisubizo cyawe gihurira n'ububyeyi, gutwita, cyangwa imyororokere y'abana
#     - Urugero: Niba ikibazo kiri kuri vitamini, sobanura uko vitamini ifasha ababyeyi batwite cyangwa abana
    
#     IMPORTANTE: Igisubizo cyawe kigomba kuba kigufi cyane (maximum 160 characters) kuko kizakoreshwa kuri USSD.
#     - Subiza mu nteruro 2-3 gusa
#     - Koresha amagambo yoroshye
#     - Ntukoreshe urutonde
#     - Tanga inama y'ibanze gusa kandi ugihurize n'ababyeyi/abana
    
#     Amakuru: {context}
#     Ikibazo: {question}
#     Igisubizo (gufi cyane, kandi gihuritswe n'ababyeyi/abana):
#     """
    
#     # Use ADK agent instead of Groq
#     response = root_agent.query(prompt_text)
#     return response


# # --- Main interaction loop with menu ---
# if __name__ == "__main__":
#     print("========= Murakaza neza kuri MotherLink Baza! =========")

#     while True:
#         print("\n1. Gukomeza kubaza | 2. Gusubira inyuma | 3. Gusohokamo")
#         choice = input("Hitamo amahitamo yawe: ")

#         if choice == "1":
#             while True:
#                 question = input("\nShyiramo ikibazo (andika '0' kugira usubire inyuma): ")
#                 if question == "0":
#                     break
                
#                 try:
#                     print("\n🔍 Tegereza...")
#                     answer = ask_motherlink(question)
#                     print("\nIgisubizo:\n", answer, "\n")
#                 except Exception as e:
#                     print(f"\n❌ Habaye ikosa: {e}\n")

#         elif choice == "2":
#             print("\nWahisemo gusubira inyuma...\n")
#             continue

#         elif choice == "3":
#             print("\nMurakoze! Muri gusohoka...")
#             break

#         else:
#             print("\nAmahitamo ntakwiye, ongera ugerageze.\n")

from google.adk.agents.llm_agent import Agent
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os

# --- Load vectorstores ---
print("Loading vectorstores...")
embedding_function = HuggingFaceEmbeddings(model_name="hkunlp/instructor-base")

maternal_store = FAISS.load_local(
    "vector_store/faiss_index",
    embeddings=embedding_function,
    allow_dangerous_deserialization=True
)
books_store = FAISS.load_local(
    "vector_store/books_faiss_index",
    embeddings=embedding_function,
    allow_dangerous_deserialization=True
)

# Combine all vectorstores
vectorstores = [maternal_store, books_store]

print("Vectorstores loaded successfully!")

# --- Prompt Template ---
template = """You are MotherLink QA assistant. Use the context below to answer clearly in Kinyarwanda if possible.
Context: {context}
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# --- Function to query vectorstores ---
def query_vectorstores(question, k=3):
    all_results = []

    for store in vectorstores:
        # Proper retriever
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})

        # Get relevant documents - FIXED
        docs = retriever.invoke(question)

        # Combine contexts but limit total tokens/characters to avoid 413 errors
        context_text = ""
        max_chars = 800  # Reduced for shorter context (was 2000)
        for d in docs:
            if len(context_text) + len(d.page_content) > max_chars:
                break
            context_text += d.page_content + "\n"

        all_results.append(context_text)

    return "\n".join(all_results)


# --- Initialize ADK Agent ---
root_agent = Agent(
    model='gemini-2.0-flash',
    name='motherlink',
    description="Motherlink.",
    instruction="""Witwa Motherlink QA. Uri umufasha ufasha ababyeyi n'abana gusubiza ibibazo bibaza ku buzima bwabo.

IMPORTANT: Tanga ibisubizo bigufi cyane (maximum 160 characters) kuko bizakoreshwa kuri USSD. 
- Subiza mu nteruro 2-3 gusa
- Koresha amagambo yoroshye
- Ntukoreshe urutonde rurerure
- Ibanze ku gisubizo cy'ibanze gusa""",
    tools=[]
)


# --- Custom query function that uses vectorstores ---
def ask_motherlink(question):
    """Query motherlink with RAG context from vectorstores"""
    
    # Get context from vectorstores
    context = query_vectorstores(question)
    
    # Prompt in Kinyarwanda
    prompt_text = f"""
    Uri umufasha wa MotherLink QA. Koresha amakuru ari hasi kugirango usubize neza mu Kinyarwanda gisesuye.
    
    IMPORTANTE: Igisubizo cyawe kigomba kuba kigufi cyane (maximum 160 characters) kuko kizakoreshwa kuri USSD.
    - Subiza mu nteruro 2-3 gusa
    - Koresha amagambo yoroshye
    - Ntukoreshe urutonde
    - Tanga inama y'ibanze gusa
    
    Amakuru: {context}
    Ikibazo: {question}
    Igisubizo (gufi cyane):
    """
    
    # Use ADK agent instead of Groq
    response = root_agent.query(prompt_text)
    return response


# --- Main interaction loop with menu ---
if __name__ == "__main__":
    print("========= Murakaza neza kuri MotherLink Baza! =========")

    while True:
        print("\n1. Gukomeza kubaza | 2. Gusubira inyuma | 3. Gusohokamo")
        choice = input("Hitamo amahitamo yawe: ")

        if choice == "1":
            while True:
                question = input("\nShyiramo ikibazo (andika '0' kugira usubire inyuma): ")
                if question == "0":
                    break
                
                try:
                    print("\n🔍 Tegereza...")
                    answer = ask_motherlink(question)
                    print("\nIgisubizo:\n", answer, "\n")
                except Exception as e:
                    print(f"\n❌ Habaye ikosa: {e}\n")

        elif choice == "2":
            print("\nWahisemo gusubira inyuma...\n")
            continue

        elif choice == "3":
            print("\nMurakoze! Muri gusohoka...")
            break

        else:
            print("\nAmahitamo ntakwiye, ongera ugerageze.\n")
