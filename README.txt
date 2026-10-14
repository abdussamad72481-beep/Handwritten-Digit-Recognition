DEEP LEARNING PROJECT: HANDWRITTEN DIGIT CLASSIFIER

What it does:
- Trains a neural network using the MNIST handwritten digit dataset.
- Predicts the digit (0 to 9) from an uploaded image.
- This is a beginner educational demo and can make mistakes.

Requirements:
- Python 3.10 or 3.11 recommended
- Internet connection for the first training run (MNIST downloads automatically)
- TensorFlow, Streamlit, Pillow, and NumPy

Run in VS Code terminal:
1. Open this folder in VS Code.
2. Install packages:
   pip install -r requirements.txt
3. Train and save the model:
   python train_model.py
4. Launch the web app:
   streamlit run app.py
5. Open the local web address printed in the terminal.
6. Upload a clear image of one handwritten digit.

Image tip:
Use a simple image with one digit centered. The app converts it to a 28x28 grayscale image similar to the MNIST training format.

Viva:
"My project is a handwritten digit recognition system using deep learning. It uses a neural network built with TensorFlow and Keras. The model is trained on MNIST images and predicts a digit from zero to nine."
