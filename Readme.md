
<div align="center">
  # SymptomIQ: AI-Powered Health Assistant
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
  [![Google Gemini](https://img.shields.io/badge/Google-Gemini%20AI-green)](https://cloud.google.com/vertex-ai)

  *An intelligent healthcare assistant that helps users predict potential health conditions based on their symptoms and provides tailored advice using Google's Gemini AI.*
</div>

## 🌟 Key Features

- 🔍 **Symptom Prediction**
  - Predict potential health conditions based on user-selected symptoms
  - TensorFlow-based machine learning model
  - Dynamic symptom filtering by category

- 🤖 **AI-Powered Advice**
  - Tailored health recommendations using Google's Gemini AI
  - Personalized responses based on predicted conditions
  - Context-aware medical insights

- 📍 **Location-Based Services**
  - Nearby doctor recommendations
  - Geolocation integration
  - Specialist matching based on conditions

- 💻 **User Experience**
  - Clean and modern web interface
  - Smooth loading animations
  - Intuitive symptom selection

## 🚀 Technologies Used

- **Frontend**: HTML, CSS, JavaScript (Vanilla JS)
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow
- **AI Integration**: Google Gemini API
- **Environment Management**: python-dotenv
- **Dependencies**: numpy, google-generativeai, flask-cors

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- Google Cloud account with Gemini API access
- TensorFlow model for symptom prediction

### Setup Steps

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/SymptomIQ.git
cd SymptomIQ
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
Create a `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Run Application**
```bash
python app.py
```

Access the web app at `http://127.0.0.1:5000`

## 📁 Project Structure

```plaintext
SymptomIQ/
├── static/                # Static files (CSS, JS, images)
│   ├── script.js         # Frontend logic
│   └── style.css         # Styling for the web app
├── templates/            # HTML templates
│   └── index.html        # Main webpage
├── models/               # Pre-trained TensorFlow model
├── .env                  # Environment variables
├── app.py               # Flask backend
├── requirements.txt      # Python dependencies
└── README.md            # Documentation
```

## 🔧 Usage

1. **Select Symptoms**
   - Use the left panel to select symptoms
   - Filter symptoms by category

2. **Get Prediction**
   - Click "Predict Health Condition" button
   - View predicted conditions

3. **AI Consultation**
   - Interact with the chatbot in the right panel
   - Receive personalized advice
   - Get doctor recommendations

4. **View Results**
   - Check predictions in the center panel
   - Review additional information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m "Add YourFeature"`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request


## 📞 Contact

- **Author**: Shri Ram Dwivedi
- **GitHub**: https://github.com/shri-ram07
- **LinkedIn**: [https://www.linkedin.com/in/shri-ram-dwivedi-8a91a3272/]()

---

<div align="center">
  Thank you for using SymptomIQ! We hope this tool helps you stay informed and take proactive steps toward better health. 🌟
</div>
