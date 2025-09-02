from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":

	# Create the path
	file_path = "./2210.03629v3.pdf"
	
	# Load the PDF
	loader = PyPDFLoader(file_path=file_path)

	# Load the document using loader
	documents = loader.load()

	# Split the text in chunks
	text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")

	# Split the documents
	documents_split = text_splitter.split_documents(documents=documents)

	# Create embeddings
	openai_embeddings = OpenAIEmbeddings()

	# Create vector store - Loaded in RAM
	vector_store = FAISS.from_documents(documents=documents_split, embedding=openai_embeddings)

	# Save the index locally
	vector_store.save_local(index_name="index", folder_path="./faiss_index_react")

	# Reload the saved index from storage
	vector_store_reloaded = FAISS.load_local(index_name="index", folder_path="faiss_index_react", embeddings=openai_embeddings, allow_dangerous_deserialization=True)

	# Pull the prompt from langchainhub
	retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

	# Create LLM
	llm = ChatOpenAI(model="gpt-4o-mini")

	# This can also be used instead of LCEL
	# # Chain to pass list of documents to the Model
	# combine_documents_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)
	# # Create retrieval chain
	# retrieval_chain = create_retrieval_chain(retriever=vector_store.as_retriever(), combine_docs_chain=combine_documents_chain)

	# Create chain using LCEL, RunnablePassthrough means pass the user input as is, without any change
	rag_chain: RunnableSequence = (
		{
			"context": vector_store.as_retriever(),
			"input": RunnablePassthrough()
		}
		| retrieval_qa_chat_prompt
		| llm
	)

	# Create the query to ask the LLM
	# Query can be directly asked, as the context is already set from the vectorstore
	query_input = """
	Create a numbered list for the data I ask you below - \
	Explain the setup used within 50 words \
	State the methods used in a comma separated style \
	The Result and Observation within 50 words
	"""

	# Invoke the chain and get the result
	result = rag_chain.invoke(input=query_input)

	# Print the result content
	print("\n", result.content, "\n")