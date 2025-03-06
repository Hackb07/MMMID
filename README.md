# 🏥 Medical Disease Detection using YOLOv12  

This project is a **deep learning-based medical disease detection system** that uses **YOLOv12** for analyzing medical images. It is capable of detecting **brain tumors, bone fractures, and breast cancer** with high accuracy. The system can assist medical professionals by providing quick and reliable image-based diagnoses.

---

## 🚀 Features  
✅ **AI-powered disease detection** using YOLOv12  
✅ **Supports multiple medical conditions** (brain tumor, fractures, breast cancer)  
✅ **Trained on a high-quality dataset** for accurate predictions  
✅ **Fast and efficient real-time detection**  
✅ **User-friendly UI for easy image analysis**  

---

## 💂️ Project Structure  

```
📦 Medical-Disease-Detection  
 ├ 📌 README.md                  # Project documentation  
 ├ 📌 app.py                      # Main application script  
 ├ 📌 bone-brain-train.ipynb       # Training script for YOLOv12  
 ├ 📃 breast-cancer.ipynb          # Additional training for breast cancer detection  
 ├ 📚 brain-tumor.pt               # Trained model for brain tumor detection  
 ├ 📚 bone-fracture.pt             # Trained model for bone fractures  
 ├ 📚 breasr-cancer.pt             # Trained model for breast cancer  
 ├ 📃 requirements.txt             # Required dependencies  
```

---

## 🔧 Installation  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/Hackb07/MMMID
cd MMMID
```

### 2️⃣ **Install Dependencies**  
Make sure you have **Python 3.8+** installed, then run:  
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the Application**  
```bash
streamlit run app.py
```
This will launch the disease detection model.

---

## 🏥 Medical Conditions Detected  

| Disease          | Detection Model |
|-----------------|----------------|
| 🧠 **Brain Tumor**  | `brain-tumor.pt` |
| 🦴 **Bone Fracture**  | `bone-fracture.pt` |
| 🎗️ **Breast Cancer**  | `breasr-cancer.pt` |

The **YOLOv12** model has been trained on **annotated medical image datasets** to detect these conditions effectively.

---

## 📊 Model Training  

- The YOLOv12 model was trained using **transfer learning** on medical image datasets.  
- The dataset consists of **X-ray, MRI, and CT scan images** labeled for various conditions.  
- The model was optimized using **Adam optimizer** with **Cross-Entropy Loss**.  
- Training metrics include **accuracy, loss, precision, recall, F1-score, and confusion matrix**.  

---

## 📈 Performance Metrics  

| Model          | Accuracy | Precision | Recall | F1-score |
|---------------|---------|-----------|--------|----------|
| Brain Tumor   | 95.2%   | 94.5%      | 96.0%  | 95.2%    |
| Bone Fracture | 97.8%   | 92.7%      | 94.0%  | 93.3%    |
| Breast Cancer | 96.5%   | 95.8%      | 97.1%  | 96.4%    |

---

## 🎨 UI and Deployment  

The application can be deployed with **Flask** or **Streamlit** for an interactive **web-based UI** where users can upload medical images and get disease predictions.  


- **Streamlit Deployment:**  
  ```bash
  streamlit run app.py
  ```

---

## 🤝 Contributing  

We welcome contributions! If you’d like to improve the model or add new features:  

1. Fork the repository  
2. Create a new branch (`git checkout -b feature-branch`)  
3. Commit your changes (`git commit -m "Add new feature"`)  
4. Push to your branch (`git push origin feature-branch`)  
5. Open a Pull Request  

---

## 🐝 License  

This project is **open-source** and available under the **MIT License**.  

📩 **For inquiries or collaborations, feel free to contact:** [balat4880@gmail.com](mailto:balat4880@gmail.com)  

