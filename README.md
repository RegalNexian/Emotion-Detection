# Emotion Detection Model

## Overview

This Emotion Detection model is designed to classify images of faces into different emotional categories. The model uses convolutional neural networks (CNNs) to analyze facial expressions and predict emotions such as anger, happiness, sadness, surprise, neutrality, disgust, and fear.

## How It Works

The model processes images through several convolutional layers, followed by pooling and fully connected layers. The images are resized to 48x48 pixels and converted to grayscale before being fed into the model. The model is trained using a dataset of labeled images, allowing it to learn the features associated with each emotion.

## Technology Stack

- **Programming Language**: Python
- **Libraries**:
  - TensorFlow (for building and training the model)
	@@ -19,77 +16,10 @@ The model processes images through several convolutional layers, followed by poo
  - scikit-learn (for data splitting)
- **Environment**: Jupyter Notebook or any Python IDE

## Requirements

To run the Emotion Detection model, ensure you have the following installed:

- Python 3.6 or higher
- Required libraries (can be installed via pip):

  ```bash
  pip install tensorflow numpy pandas opencv-python scikit-learn

```bash
- A compatible IDE or Jupyter Notebook for running the code.
## How to Use
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd emotion-detection
   ```bash
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```bash
3. Prepare your training and testing datasets in the specified folder structure:
   ```bash

   train/
       angry/
       happy/
       sad/
       surprised/
       neutral/
       disgust/
       fear/
   test/
       angry/
       happy/
       sad/
       surprised/
       neutral/
       disgust/
       fear/

   ```

4. Update the paths in `detect.py` to point to your training and testing datasets.
5. Run the model:

   ```bash
   ```

6. The model will save the best weights to the specified path after training.

## Model Architecture

- **Input Layer**: Accepts images of shape (48, 48, 1)
- **Convolutional Layers**: Four convolutional layers with batch normalization and max pooling
- **Fully Connected Layers**: One dense layer with dropout for regularization, followed by an output layer with softmax activation for multi-class classification

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Colab Notebook

You can run this model directly in Google Colab using [Link Here](https://colab.research.google.com/drive/1Vihs2DNeTemnSQVTH8NDoRMNe4RYtDmr?usp=sharing).
