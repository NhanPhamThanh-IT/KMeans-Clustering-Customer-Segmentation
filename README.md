# <div align="center">K-Means Clustering Customer Segmentation</div>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-Enabled-brightgreen.svg" alt="Streamlit"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-Model-orange.svg" alt="scikit-learn"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation/stargazers"><img src="https://img.shields.io/github/stars/NhanPhamThanh-IT/KMeans-Clustering-Customer-Segmentation?style=social" alt="GitHub stars"></a>
</p>

<div align="justify">

## Overview

This project demonstrates customer segmentation using the K-Means clustering algorithm. It provides an interactive web application (built with Streamlit) that predicts customer segments based on annual income and spending score, using a model trained on the popular Mall Customers dataset.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [K-Means Clustering](#k-means-clustering)
4. [Model Training](#model-training)
5. [Web Application](#web-application)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Advanced Usage](#advanced-usage)
9. [Requirements](#requirements)
10. [Contributing](#contributing)
11. [FAQ](#faq)
12. [References](#references)
13. [License](#license)

---

## Project Structure

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

## Dataset

- **Source:** `dataset/mall_customers.csv`
- **Description:** Contains 200 customer records from a mall, with the following columns:
  - `CustomerID`: Unique identifier
  - `Gender`: Male/Female
  - `Age`: Customer age
  - `Annual Income (k$)`: Annual income in thousands of dollars
  - `Spending Score (1-100)`: Score assigned by the mall based on customer behavior and spending

The model uses only `Annual Income (k$)` and `Spending Score (1-100)` for clustering.

---

## K-Means Clustering

K-Means is an unsupervised machine learning algorithm that partitions data into `k` clusters, where each data point belongs to the cluster with the nearest mean. The algorithm iteratively updates cluster centroids and assignments to minimize within-cluster variance.

- **Why K-Means?**
  - Simple and efficient for large datasets
  - Well-suited for customer segmentation based on numerical features

For more details, see [`docs/kmeans-clustering.md`](docs/kmeans-clustering.md).

---

## Model Training

The model is trained in the Jupyter notebook [`model/model_training.ipynb`](model/model_training.ipynb):

1. **Data Loading:** Reads the dataset and selects relevant features.
2. **Preprocessing:** Checks for missing values and explores the data.
3. **Choosing Clusters:** Uses the Elbow Method to determine the optimal number of clusters (`k=5`).
4. **Training:** Fits a KMeans model on `Annual Income` and `Spending Score`.
5. **Saving Model:** Serializes the trained model to `model/model.pkl` using pickle.

---

## Web Application

The interactive app is built with Streamlit and consists of:

- **Input:** Users enter annual income and spending score.
- **Prediction:** The app predicts the customer segment (cluster) using the trained model.
- **Output:** Displays the predicted segment.

**Main files:**
- `app/main.py`: Entry point, initializes model and UI.
- `app/model.py`: Loads the trained model and provides prediction.
- `app/ui.py`: Streamlit UI for user interaction.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd KMeans-Clustering-Customer-Segmentation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run app/main.py
   ```

2. **Open your browser:**  
   Visit `http://localhost:8501` to access the app.

3. **Interact:**  
   - Enter values for "Annual Income (k$)" and "Spending Score (1-100)".
   - Click "Predict" to see the predicted customer segment.

---

## Advanced Usage

- **Retrain the Model:**
  - Open `model/model_training.ipynb` in Jupyter Notebook.
  - Modify the code or dataset as needed.
  - Run all cells to retrain and save a new model to `model/model.pkl`.
  - Restart the Streamlit app to use the updated model.

- **Custom Features:**
  - You can extend the model to use more features (e.g., Age, Gender) by modifying the notebook and updating the app code accordingly.

- **Deployment:**
  - Deploy the app to [Streamlit Cloud](https://streamlit.io/cloud), Heroku, or any cloud provider supporting Python and Streamlit.
  - For Docker deployment, create a `Dockerfile` as described in [`docs/streamlit.md`](docs/streamlit.md).

---

## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - streamlit
  - jupyter (optional, for training notebook)

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your code follows best practices and is well-documented.

---

## FAQ

**Q: Can I use a different dataset?**
A: Yes! Replace `dataset/mall_customers.csv` with your own data and retrain the model using the notebook.

**Q: How do I add more features to the clustering?**
A: Update the feature selection in `model/model_training.ipynb` and adjust the app code to accept new inputs.

**Q: The app doesn't start or throws an error. What should I do?**
A: Ensure all dependencies are installed and you are using a compatible Python version. Check the error message for details.

**Q: How do I deploy this app online?**
A: See the deployment section above or refer to [`docs/streamlit.md`](docs/streamlit.md) for detailed instructions.

---

## References

- [Mall Customers Dataset on Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [K-Means Clustering Guide](docs/kmeans-clustering.md)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Project Dataset Documentation](docs/dataset.md)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Demo

Below is a preview of the customer segmentation app:

![App Screenshot](docs/demo-screenshot.png)

*Figure: Example of the Streamlit interface for customer segmentation.*

---

## Features

- Interactive web interface for customer segmentation
- Real-time prediction of customer clusters
- Visualizations of clusters and data distribution
- Easy retraining with new data or features
- Modular codebase for easy customization

---

## How It Works

1. **User Input:** Enter annual income and spending score.
2. **Model Prediction:** The trained KMeans model assigns the customer to a segment.
3. **Result Display:** The app shows the predicted segment and (optionally) a visualization of all clusters.

---

## Visualization Examples

The notebook and app provide visualizations such as:

- Elbow Method plot for optimal cluster selection
- 2D scatter plot of customer segments

![Cluster Visualization](docs/cluster-visualization.png)

---

## Troubleshooting

- **ModuleNotFoundError:** Ensure all dependencies are installed with `pip install -r requirements.txt`.
- **Streamlit not launching:** Check your Python version and that Streamlit is installed in your environment.
- **Model not found:** Retrain the model using the notebook if `model/model.pkl` is missing.

---

## Contact

For questions, suggestions, or support, please open an issue or contact [ptnhanit230104@gmail.com](mailto:ptnhanit230104@gmail.com).

---

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Kaggle Mall Customers Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)

---

</div>

<div align="center">

**For more information, see the documentation in the `docs/` folder.**

</div>