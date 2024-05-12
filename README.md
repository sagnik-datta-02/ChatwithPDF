# RAG-based Chat with PDF

This project enables conversational interaction with PDF documents using the Retrieval-Augmented Generation (RAG) model. Users can ask questions related to the content of uploaded PDF files, and the system will provide detailed responses using RAG-based conversation generation techniques.

## Features

- **PDF Upload**: Users can upload one or multiple PDF files containing the information they want to inquire about.

- **Text Extraction**: Extracts text content from uploaded PDF files for processing and analysis.

- **Text Chunking**: Splits the extracted text into smaller chunks for efficient processing and retrieval.

- **Vector Store Creation**: Utilizes FAISS to create a vector store from the text chunks, enabling fast and accurate retrieval of relevant information.

- **Conversational Interface**: Utilizes the RAG model to generate responses to user queries in a conversational manner.

## Tech Stack
- **Python**: Programming language used for development.
- **Streamlit**: Web application framework for building interactive web applications.
- **PyPDF2**: Python library for reading PDF files.
- **Langchain**: Framework for developing RAG model using LLM.
- **FAISS**: Library for efficient similarity search and clustering of dense vectors.
- **Anthropic**: API for claude-3-sonnet-20240229 LLM.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/sagnik-datta-02/ChatwithPDF.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

   - Ensure you have an Anthropic API key and set it in your environment variables as `ANTHROPIC_API_KEY`.
   
   - Make sure to have a `.env` file containing your environment variables, including the Anthropic API key.

4. Run the Streamlit app:

```bash
streamlit run multipdfragapp.py
```

## Usage

1. Upload PDF files containing the information you want to inquire about.
2. Click on "Submit & Process" to process the uploaded PDF files.
3. Ask a question related to the content of the uploaded PDF files in the text input field and receive a response to your question.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://github.com/mstamy2/PyPDF2)
- [langchain](https://github.com/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Anthropic](https://anthropic.com/)
- [dotenv](https://github.com/theskumar/python-dotenv)
