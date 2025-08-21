This is a simple Q & A based RAG application where the dataset is Alice in Wonderland.txt file.

The vector database used here is Chroma, LLM used is Gemini Flash 2.5 and embeddings from Hugging face

Note: For accessing the LLM use your own LLM key for Gemini Flash 2.5

Steps to run this project on your machine
1. Run the requirements.txt file : pip -r requirements.txt
2. Run the main.py file : uvicorn server:app --reload
3. Run the streamlit UI: streamlit run app.py

