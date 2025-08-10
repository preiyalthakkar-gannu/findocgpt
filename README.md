# FinDocGPT 
FinDocGPT
FinDocGPT is a Financial Intelligence Copilot that processes financial documents (PDF,
DOCX, TXT), performs QA, sentiment analysis, forecasting, and strategy suggestions to aid
in decision-making.

Features
  - Document Q&A: Ask questions and get precise answers from your uploaded financial
  documents.
  - Sentiment Analysis: Understand the tone of financial news or reports.
  - Forecasting: Get data-driven growth or decline predictions.
  - Strategy Suggestions: Actionable recommendations based on analysis.
  - Export Options: Export processed data and reports to PDF.
    
Installation
  1. Clone the repository:
     git clone https://github.com/preiyalthakkar-gannu/findocgpt.git
  2. Navigate to the project folder:
  cd findocgpt
  3. Create and activate a virtual environment:
     python -m venv .venv
     .venv\Scripts\activate (Windows)
  4. Install dependencies:
     pip install -r requirements.txt
   
Usage
  1. Activate your virtual environment.
  2. Run the app:
     streamlit run app.py
  3. Upload your financial document (PDF/DOCX/TXT).
  4. Use the tabs for Q&A, Sentiment, Forecast, and Strategy.
   
Project Structure
  - core/ - Main processing and logic modules.
  - nltk_data/sentiment/ - Sentiment model data.
  - samples/ - Example financial documents.
  - app.py - Streamlit application entry point.
  - config.yaml - Configuration file.
  - requirements.txt - Python dependencies.

License
  This project is licensed under the MIT License - see the LICENSE file for details.
