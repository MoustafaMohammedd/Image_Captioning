# Image Captioning 🚀

This project implements an image captioning system using PyTorch, leveraging a ResNet-50 encoder and an LSTM-based decoder. The model is trained and evaluated on the Flickr-8k dataset.

## 📦Project Structure 

```
.
├── main.py
├── requirements.txt
├── README.md
├── best_model_lstm/   # Model checkpoint (not uploaded)
├── config/
│   ├── config.py
│   └── __init__.py
├── data/
│   └── download_data.py
├── flickr-8k-images-with-captions/   # Only 5 images uploaded
│   ├── captions.txt
│   └── images/
├── images/
├── notebooks/           # Jupyter notebooks for experimentation and analysis
├── runs/                # For logging by tensorboard
│   ├── __init__.py
│   ├── data.py
│   ├── inference.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
└── test_examples/
└── test_images_output/
```
## Test Images

Below are some sample outputs from the model:

![Output 1](test_images_output/Screenshot%202025-06-09%20142718.png)
![Output 2](test_images_output/Screenshot%202025-06-10%20140543.png)
![Output 3](test_images_output/Screenshot%202025-06-10%20140604.png)
![Output 4](test_images_output/Screenshot%202025-06-10%20140621.png)
![Output 5](test_images_output/Screenshot%202025-06-10%20140642.png)


## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   - Run the following script to download the Flickr-8k dataset:
     ```
     python data/download_data.py
     ```
   - Ensure the images and captions are extracted to `flickr-8k-images-with-captions/`.

## Training

To train the model, run:

```
python src/train.py
```

- Training checkpoints and the best model will be saved in the `best_model_lstm/` directory.
- Training logs and images are saved for TensorBoard visualization.

## Inference

To generate captions for new images, run:

```
python src/inference.py
```

Edit the `img_dir` variable in `src/inference.py` to point to your test image.

## Configuration

Model and training hyperparameters are defined in [`config/config.py`](config/config.py).

## Utilities

- Data preprocessing and augmentation: [`src/utils.py`](src/utils.py)
- Dataset and DataLoader: [`src/data.py`](src/data.py)
- Model definition: [`src/models.py`](src/models.py)
- Training loop: [`src/train.py`](src/train.py)
- Inference: [`src/inference.py`](src/inference.py)

## Requirements

See [`requirements.txt`](requirements.txt) for all dependencies.

## Example

After training, you can test the model on sample images in `test_examples/` or your own images.

---

**Note:**  
- Make sure you have a GPU available for faster training.
- Adjust paths in the code if your directory structure differs.
