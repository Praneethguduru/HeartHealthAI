
# ❤️ Heart Health Prediction Model

This project uses machine learning to predict the risk of heart disease based on user health data. The model is trained on the **Framingham Heart Study dataset**, a widely used dataset for cardiovascular health analysis.

## 🚀 Features

- Predicts risk of heart disease based on user inputs
- Built using **Python** and **Scikit-learn**
- Interactive **Gradio** web interface for easy testing
- Backend integration with **MySQL** for data storage
- (Optional) Frontend built with **Angular** for a complete web application

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Gradio
- MySQL
- Angular (Frontend)

---

## 📂 Project Structure

```
heart-health-prediction/
├── model/                 # ML model training & evaluation
│   ├── train_model.py
│   └── model.pkl
├── app/                   # Backend with Gradio interface
│   └── app.py
├── frontend/              # Angular frontend (optional)
│   └── src/...
├── database/              # MySQL setup scripts
│   └── setup.sql
├── README.md
└── requirements.txt
```

---

## 🧠 How It Works

1. The model is trained using the Framingham dataset.
2. User inputs health information via the Gradio interface or web frontend.
3. The model predicts the risk of heart disease.
4. Data is stored in a MySQL database for future reference (optional).

---

## ⚡ Quick Start

### Backend (Gradio App)

```bash
cd app
pip install -r requirements.txt
python app.py
```

### Frontend (Optional - Angular)

```bash
cd frontend
npm install
ng serve
```

---

## 📊 Dataset

The project uses the **Framingham Heart Study dataset**, which contains information such as:

- Age
- Gender
- Blood pressure
- Cholesterol
- Smoking habits
- Diabetes status
- And more...

---

## 📝 Future Improvements

- Enhance model accuracy with hyperparameter tuning
- Add authentication for secure access
- Deploy on cloud platforms
- Improve frontend UI/UX

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests.

---

## 📫 Contact

For any queries, reach out to me:

**Guduru Praneeth**  
Email: *[Your Email Here]*  
LinkedIn: *[Your LinkedIn Here]*  
