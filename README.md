# Multi-LLM Collaboration Chat
This is a Gradio app that enables multi-LLM collaboration chat. It uses the Hugging Face Hub to interact with different language models and allows users to upload files and save conversation history.

## Features
- Multi-LLM collaboration chat
- File upload (PDF, DOCX, XLSX)
- Conversation history saving
- Support for multiple language models

## Requirements
- gradio>=4.0.0
- requests>=2.31.0
- huggingface_hub
- pdfplumber
- python-docx
- openpyxl

## Usage
1. Install the required dependencies by running `pip install -r requirements.txt`
2. Run the app by executing `python app.py`
3. Open a web browser and navigate to the app's URL (usually `http://localhost:7860`)
4. Start chatting with the language models and upload files as needed

## Notes
- This app is designed for research and development purposes only
- The app's functionality and performance may vary depending on the language models used and the system's hardware specifications
