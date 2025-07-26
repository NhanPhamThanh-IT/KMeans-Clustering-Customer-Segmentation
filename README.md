# <div align="center">K-Means Clustering Customer Segmentation</div>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-Enabled-brightgreen.svg" alt="Streamlit"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-Model-orange.svg" alt="scikit-learn"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation/stargazers"><img src="https://img.shields.io/github/stars/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation?style=social" alt="GitHub stars"></a>
  <a href="https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation/issues"><img src="https://img.shields.io/github/issues/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation?color=important" alt="Open Issues"></a>
  <a href="https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation/network"><img src="https://img.shields.io/github/forks/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation?style=social" alt="GitHub forks"></a>
  <!-- Add more badges as needed -->
</p>

<div align="justify">

<p>K-Means Clustering Customer Segmentation is a user-friendly, interactive web application built with Python, scikit-learn, and Streamlit. It enables businesses and data enthusiasts to segment customers based on annual income and spending score using the K-Means clustering algorithm. The app provides real-time predictions, clear visualizations, and supports easy retraining with new data or features. Ideal for marketing, retail, banking, and more, it helps identify high-value or at-risk customer groups, personalize offers, and optimize business strategies. The modular codebase and comprehensive documentation make it easy to customize, extend, and deploy in various environments.</p>

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Business Use Cases](#-business-use-cases)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Project Structure](#️-project-structure)
- [Technical Architecture](#-technical-architecture)
- [Dataset](#-dataset)
- [K-Means Clustering](#-k-means-clustering)
- [How It Works: Step-by-Step](#️-how-it-works-step-by-step)
- [Behind the Scenes: Code Structure](#️-behind-the-scenes-code-structure)
- [Customization & Extensibility](#️-customization--extensibility)
- [Sample Input/Output](#-sample-inputoutput)
- [Installation & Requirements](#-installation--requirements)
- [Usage](#-usage)
- [Advanced Usage](#-advanced-usage)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Security & Privacy](#-security--privacy)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [FAQ](#-faq)
- [Support](#-support)
- [Community & Social](#-community--social)
- [Changelog](#-changelog)
- [Roadmap](#-roadmap)
- [Glossary](#-glossary)
- [References & Acknowledgements](#-references--acknowledgements)
- [License](#-license)
- [Citation](#-citation)

---

## 🚀 Overview

**K-Means Clustering Customer Segmentation** is an end-to-end, interactive web application for segmenting customers using unsupervised machine learning. Built with Python, scikit-learn, and Streamlit, this project enables businesses and data enthusiasts to:

- **Identify distinct customer groups** based on spending patterns and income
- **Visualize clusters** for actionable business insights
- **Experiment with new data** and retrain models easily

**Business Value:**
- Target marketing campaigns to specific customer segments
- Personalize offers and improve customer retention
- Discover high-value or at-risk customer groups

---

## 💼 Business Use Cases

- **Retail:** Segment shoppers to tailor promotions and loyalty programs.
- **Banking:** Identify high-value clients for premium services.
- **E-commerce:** Personalize recommendations and offers.
- **Hospitality:** Group guests for targeted experiences.
- **Telecom:** Detect churn-prone customers and upsell opportunities.
- **Education:** Cluster students for personalized learning paths.

---

## 📦 Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd KMeans-Clustering-Customer-Segmentation
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the app:**
   ```bash
   streamlit run app/main.py
   ```
4. **Open your browser:**
   Visit [http://localhost:8501](http://localhost:8501)

---

## ✨ Features

| Feature                        | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| Interactive Web UI             | User-friendly Streamlit interface for input and results           |
| Real-time Prediction           | Instantly predicts customer segment from input values             |
| Visualizations                 | Cluster plots, Elbow method, and more (add your screenshots!)     |
| Easy Retraining                | Jupyter notebook for model retraining with new data/features      |
| Modular Codebase               | Clean separation of UI, model, and logic for easy customization   |
| Deployment Ready               | Simple to deploy on Streamlit Cloud, Heroku, or Docker            |
| Documentation                  | Extensive docs for dataset, clustering, and deployment            |

---

## 🗂️ Project Structure

```
KMeans-Clustering-Customer-Segmentation/
  ├── app/
  │   ├── main.py         # Streamlit app entry point
  │   ├── model.py        # Model loading and prediction logic
  │   └── ui.py           # Streamlit UI components
  ├── dataset/
  │   └── mall_customers.csv  # Customer data
  ├── model/
  │   ├── model_training.ipynb # Jupyter notebook for training
  │   └── model.pkl            # Trained KMeans model
  ├── docs/
  │   ├── dataset.md
  │   ├── kmeans-clustering.md
  │   └── streamlit.md
  ├── requirements.txt
  └── README.md
```

---

## 🏗️ Technical Architecture

```mermaid
flowchart TD
    A[User Input (Streamlit UI)] --> B[Model Loader (app/model.py)]
    B --> C[Trained KMeans Model (model/model.pkl)]
    A --> D[UI Logic (app/ui.py)]
    B --> E[Prediction Output]
    D --> E
    E --> F[Visualization (matplotlib/seaborn)]
    F --> G[Display Results in Streamlit]
    subgraph Data Science
        C
        F
    end
```

---

## 📊 Dataset

- **File:** `dataset/mall_customers.csv`
- **Source:** [Kaggle Mall Customers Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Columns:**
  - `CustomerID`: Unique identifier
  - `Gender`: Male/Female
  - `Age`: Customer age
  - `Annual Income (k$)`: Annual income in thousands of dollars
  - `Spending Score (1-100)`: Score assigned by the mall based on customer behavior and spending

*Note: The default model uses only `Annual Income (k$)` and `Spending Score (1-100)` for clustering.*

---

## 🤖 K-Means Clustering

K-Means is an unsupervised algorithm that partitions data into `k` clusters, grouping similar data points together. It is widely used for customer segmentation due to its simplicity and effectiveness.

- **How it works:**
  1. Choose `k` cluster centers (centroids)
  2. Assign each data point to the nearest centroid
  3. Update centroids as the mean of assigned points
  4. Repeat until assignments stabilize

- **Why K-Means?**
  - Fast and scalable
  - Intuitive results for business users
  - Well-suited for numerical features

For more, see [`docs/kmeans-clustering.md`](docs/kmeans-clustering.md).

---

## 🏗️ How It Works: Step-by-Step

1. **Data Preparation:**
   - Load and explore the dataset
   - Select relevant features (default: income & spending score)
2. **Model Training:**
   - Use the Elbow Method to find optimal `k`
   - Train KMeans on selected features
   - Save the trained model as `model/model.pkl`
3. **Web Application:**
   - User enters income and spending score
   - App loads the trained model and predicts the segment
   - Results and (optionally) cluster visualizations are displayed

---

## 🖥️ Behind the Scenes: Code Structure

- `app/main.py`: Streamlit entry point; initializes app, loads model, and handles routing
- `app/model.py`: Handles model loading and prediction logic
- `app/ui.py`: Contains Streamlit UI components for input and output
- `model/model_training.ipynb`: Jupyter notebook for data exploration, training, and saving the model

---

## 🛠️ Customization & Extensibility

- **Add More Features:**
  - Edit `model/model_training.ipynb` to include more columns (e.g., Age, Gender)
  - Update the app UI in `app/ui.py` to accept new inputs
- **Use Your Own Data:**
  - Replace `dataset/mall_customers.csv` with your dataset (same or similar format)
  - Retrain the model using the notebook
- **Change Number of Clusters:**
  - Adjust `k` in the notebook and retrain
- **Deploy Anywhere:**
  - See [`docs/streamlit.md`](docs/streamlit.md) for deployment guides (Streamlit Cloud, Docker, etc.)

---

## 📝 Sample Input/Output

**Sample Input:**
- Annual Income (k$): `60`
- Spending Score (1-100): `42`

**Sample Output:**
```
Predicted Segment: 3
This customer belongs to the "Average Income, Average Spending" group.
```

---

## 📥 Installation & Requirements

- **Python:** 3.7 or higher
- **Install all dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **requirements.txt includes:**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - streamlit
  - jupyter, ipykernel (optional, for notebook)

---

## 🧑‍💻 Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app/main.py
   ```
2. **Open your browser:**
   Go to [http://localhost:8501](http://localhost:8501)
3. **Interact:**
   - Enter "Annual Income (k$)" and "Spending Score (1-100)"
   - Click "Predict" to see the customer segment

---

## 🔬 Advanced Usage

- **Retrain the Model:**
  - Open `model/model_training.ipynb` in Jupyter
  - Modify code or data as needed
  - Run all cells to retrain and save a new model
  - Restart the app to use the updated model
- **Deploy Online:**
  - See [`docs/streamlit.md`](docs/streamlit.md) for deployment instructions

---

## 🛠️ Troubleshooting

| Problem                        | Solution                                                      |
|-------------------------------|---------------------------------------------------------------|
| `ModuleNotFoundError`          | Run `pip install -r requirements.txt`                         |
| Streamlit not launching        | Check Python version and Streamlit installation               |
| Model file not found           | Retrain model using the notebook                              |
| Port 8501 already in use       | Use `streamlit run app/main.py --server.port <other_port>`    |
| UI not updating after retrain  | Restart Streamlit app                                         |

---

## 🏅 Best Practices

- Always explore your data before training
- Use the Elbow Method to select the best `k`
- Document any changes to the dataset or features
- Test the app after retraining the model
- Use virtual environments for dependency management
- Add screenshots to the README for better engagement

---

## 🔒 Security & Privacy

- **No personal data is stored** by the app; all predictions are in-memory
- If using real customer data, ensure compliance with GDPR or local privacy laws
- Do not upload sensitive data to public repositories

---

## 📝 Documentation

- [`docs/dataset.md`](docs/dataset.md): Dataset details and schema
- [`docs/kmeans-clustering.md`](docs/kmeans-clustering.md): K-Means theory and implementation
- [`docs/streamlit.md`](docs/streamlit.md): Streamlit and deployment guides

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

**Best Practices:**
- Write clear, concise commit messages
- Add docstrings and comments
- Test your code before submitting

---

## ❓ FAQ

**Q: Can I use a different dataset?**  
A: Yes! Replace `dataset/mall_customers.csv` and retrain the model.

**Q: How do I add more features?**  
A: Update feature selection in the notebook and app UI.

**Q: The app doesn't start or throws an error. What should I do?**  
A: Ensure all dependencies are installed and Python version is compatible. Check error messages for details.

**Q: How do I deploy this app online?**  
A: See [`docs/streamlit.md`](docs/streamlit.md) for deployment instructions.

---

## 🛟 Support

- Open an [issue](https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation/issues) for bugs or feature requests
- Email: [ptnhanit230104@gmail.com](mailto:ptnhanit230104@gmail.com)

---

## 🌐 Community & Social

- [Discussions](https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation/discussions) (ask questions, share ideas)
- [Contributors](https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation/graphs/contributors)
- Suggest a Slack/Discord channel for real-time help!

---

## 🗒️ Changelog

- **v1.0**: Initial release with Streamlit app, model training notebook, and documentation
- **v1.1**: Improved modularity, added advanced usage and deployment docs
- **v1.2**: Enhanced README, added FAQ and troubleshooting

---

## 🚀 Roadmap

- [ ] Add more clustering algorithms (DBSCAN, Hierarchical)
- [ ] Add user authentication for private deployments
- [ ] Enable export of cluster assignments
- [ ] Add more visualizations (3D plots, interactive charts)
- [ ] Docker Compose for multi-service deployment
- [ ] Add REST API for programmatic access
- [ ] Internationalization (i18n) support

---

## 📖 Glossary

- **K-Means:** Unsupervised clustering algorithm
- **Cluster:** Group of similar data points
- **Centroid:** Center of a cluster
- **Elbow Method:** Technique to find optimal number of clusters
- **Streamlit:** Python library for building web apps
- **scikit-learn:** Python ML library

---

## 📚 References & Acknowledgements

- [Mall Customers Dataset on Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Project Dataset Documentation](docs/dataset.md)

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this project in your research, please cite as:

```
@misc{KMeansClusteringCustomerSegmentation,
  author = {Nhan Pham Thanh},
  title = {K-Means Clustering Customer Segmentation},
  year = {2024},
  howpublished = {\url{https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation}}
}
```

</div>

<div align="center">

**For more information, see the documentation in the `docs/` folder.**

*Add your screenshots to the `docs/` folder and reference them above for a more visual README!*

</div>