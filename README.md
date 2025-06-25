# 📚 EduPersona: Personal Learning Assistant

EduPersona is an AI-powered personal learning assistant designed to help users understand and interact with PDF content through quizzes, theoretical Q&A, summaries, and chat. It integrates advanced generative AI models (Gemini & OpenAI) with a user-friendly Streamlit interface.

---

## 🚀 Features

- 📄 **PDF Upload**: Upload and process academic materials for further learning activities.
- 🧠 **Quiz Generator**: Automatically generate multiple-choice questions from uploaded PDF content using Gemini Pro.
- ✅ **Take Quiz**: Attempt generated quizzes and receive feedback with simplified explanations on wrong answers.
- 📖 **Theory Q&A**: Generate theoretical questions, write answers, and receive AI-based evaluation.
- 📝 **PDF Summarizer**: Generate detailed summaries (up to 7 pages) from long PDFs.
- 💬 **Chat with PDF**: View and chat with uploaded PDF content using LangChain's document handling capabilities.
- 👤 **Authentication**: Secure login and signup system using SQLite and hashed passwords.

---

## 🧰 Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python
- **AI Models**: Google Gemini API (via Vertex AI), OpenAI (via LangChain)
- **Document Handling**: PyPDFLoader, pdfminer
- **Database**: SQLite
- **Other Libraries**: 
  - `dotenv` for environment variable management  
  - `fpdf` for PDF generation  
  - `streamlit_pdf_viewer` for in-app PDF rendering  
  - `werkzeug.security` for password hashing

---

## 📂 Folder Structure

```
EduPersona/
│
├── main/                             # Main Streamlit app and logic
│   ├── app.py                        # Streamlit application logic
│   ├── users.db                      # SQLite user database
│   └── notes/                        # Saved PDFs and summaries
│
├── .env                              # API keys and secrets
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/naveen-dsatm/EduPersona.git
   cd EduPersona
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add:
   ```
   API_KEY=your_google_gemini_api_key
   ```

5. **Add Vertex AI Credentials**:
   Download your Google Cloud service account key and set the path in your script:
   ```python
   os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"
   ```

6. **Run the App**:
   ```bash
   streamlit run main/app.py
   ```

---


## 📌 Future Improvements

- Chat memory with context history
- Support for DOCX/text files
- User dashboard with learning progress
- Email-based account verification
- Better error handling for model output

---

## 🧑‍💻 Author

**Naveen S** – [naveen-dsatm on GitHub](https://github.com/naveen-dsatm)

---

## 🪪 License

This project is for educational and research purposes only.
