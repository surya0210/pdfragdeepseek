**Bujji-For PDF**
Advanced RAG System
This project implements an A-RAG system for PDF documents using LangChain, PyPDF, and Gradio. Upload PDFs, ask questions, and receive answers with relevant sources.

-----------------------------------------------------------------------------------------------------------------------------------------------
**Steps:**

1.Install Ollama https://ollama.com/download/windows

2.After installing Open ollama in Powershell/CMD and start the deepseek instance 
  
  ![image](https://github.com/user-attachments/assets/9ac2a44d-e938-4a6b-bf37-9aeea7669b4e)
  This step will take time depening on the system's memory specifications.

3.You can close the engine and can write serve it using ollama server 
If encountering errors kill PID by finding using 

![image](https://github.com/user-attachments/assets/8c2c5ac2-0d12-4348-ae9b-bc1cb56cfb46)

and kill the process if not necessary using the pid.

----------------------------------------------------------------------------------------------------------------------------------------------

Requirements:

Python 3.8+
Required packages:
  langchain
  langchain_community
  langchain_ollama
  pypdf
  gradio/streamlit
-----------------------------------------------------------------------------------------------------------------------------------------------
Constraints:
  Temperature ->0.3

and run the model using **python bujji.py**


if you don't want thinking part you can use regex and remove the thinking part.

Happy Coding..

