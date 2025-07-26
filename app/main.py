"""main.py

Entry point for the Streamlit app.
"""

from model import CustomerSegmentModel
from ui import CustomerSegmentationApp


def main():
    """Run the customer segmentation app."""
    model = CustomerSegmentModel(model_path="model/model.pkl")
    app = CustomerSegmentationApp(model)
    app.run()


if __name__ == "__main__":
    main()
