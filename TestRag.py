
# Two modes to run this file.
# mode=setup : To setup the vector database with document embeddings.
# mode=query : To query the vector database with a question.

# setup mode steps:
# Read File.
# Convert into chunks with 500 size and overlap of 50.
# Create embeddings using HuggingFaceEmbeddings.
# Store in ChromaDB vector database.



import sys

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

print ("Starting the script...")
persist_directory = r"C:\Users\sravani k\git\aiprograms\chroma_db"
# Parse arguments to choose mode
mode = sys.argv[1] if len(sys.argv) > 1 else "setup"
if mode == "setup":
    print ("Running in setup mode...")
    # Load the document
    loader = TextLoader(r"C:\Users\sravani k\Downloads\data.txt", "utf-8")
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # Store in ChromaDB vector database
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="india_collection",
                                        persist_directory=persist_directory)
    print ("Vector database setup complete.")

    # Save the vectorstore to disk to the persist directory
    vectorstore.add_documents(docs)
    print ("Vector database persisted to disk.")

# query mode steps:
# Load the vector database.
# Create a RetrievalQA chain with ChatOpenAI and the vector database as retriever.
# Ask a question and get the answer.

elif mode == "query":
    print ("Running in query mode...")
    # Load the vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="india_collection", embedding_function=embeddings,
                         persist_directory=persist_directory)


    # Ask a question
    question1 = "What is the capital of India?"
    answer = vectorstore.similarity_search(question1, k=3)
    print("\nTop Results:")
    for i, doc in enumerate(answer, 1):
        print(f"Chunk {i}:", doc.page_content[:400], "...\n")



    print("Loading Local HuggingFace Model... (Takes time once)")
    # model_id = "HuggingFaceH4/zephyr-7b-beta"
    model_id = "google/flan-t5-base"  # Only ~1GB
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"  # alternative

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Create RetrievalQA chain
    # 🔹 Strong anti-hallucination RAG prompt
    prompt_template = """
    You are a subject teacher.

    Use ONLY the information provided in the context below.
    If the answer is not found in the context, reply EXACTLY: "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.run(question1)
    print("\n🚀 Final Answer:", result)
