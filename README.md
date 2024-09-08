                              ****Study Helper: Question Answer Chatbot****

This project is a Study Helper application built with Streamlit. It allows users to upload documents (PDF, DOCX, PPTX, DOC, PPT, and TXT) and then provides several functionalities, including summarization, topic-based summaries, concept explanations, custom quiz generation, and answering questions based on the content of the uploaded document.

Features:
        Document Upload: Upload a document in formats such as PDF, DOCX, PPTX, DOC, PPT, or TXT.
        Text Extraction: Automatically extract text from the uploaded document.
        Summarization: Generate a summary of the extracted text.
        Topic-Based Summarization: Summarize the text based on user-specified topics.
        Concept Explanation: Explain concepts by integrating document content with web search results.
        Custom Quiz Generation: Create quiz questions based on a user-specified topic.
        Question Answering: Answer questions based on the content of the document.
        
Requirements:
        Python 3.7+
        Streamlit
        Google Generative AI
        PyMuPDF (Fitz)
        python-docx
        python-pptx
        comtypes
        pythoncom
        requests

        
Usage:
        Run the Streamlit App:  streamlit run app.py
        Upload Your Document:  Use the file uploader to select and upload a document in one of the supported formats.
        Summarize the Document:  A summary of the document will be generated automatically.
        Topic-Based Summarization:  Enter topics in the input box, separated by commas, and click the "Summarize Based on Topics" button to get a topic-focused summary.
        Explain Concepts:  Enter a concept in the input box and click "Explain Concept" to get an explanation based on the document and additional web search results.
        Generate a Custom Quiz:  Enter a topic and click "Generate Custom Quiz" to create quiz questions based on the document's content.
        Ask a Question:  Enter a question in the input box to get an answer based on the document.

        
Deploying on Streamlit Community Cloud:
        You can deploy the app on Streamlit Community Cloud
        Push your code to GitHub.
        Navigate to Streamlit Community Cloud and link your GitHub repository.
        Click "Deploy" to make your app live.

        
Contributing:
        Contributions are welcome! Please fork the repository and submit a pull request with your changes.
