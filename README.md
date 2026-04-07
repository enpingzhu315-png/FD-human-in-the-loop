# Multi-Modal Fault Diagnosis Model

This project implements a multi-modal fault diagnosis system based on both image and text data. It includes a text feature network, an image feature network, a fusion feature network, and a multi-source input feature judgment network. The project is built on the PyTorch framework and supports GPU acceleration for training and inference.

## Directory Structure

Assuming the repository root is `MultiModalFault`, the recommended structure is:

```
MultiModalFault/
│
├─ data/                   # Raw data folder
│   ├─ train_L/            # Original training images
│   └─ multi_data_0.05.json # Original annotation JSON file
│
├─ result/                 # Folder to save training results and outputs
│   └─ CASE1/              # First experiment results
│       ├─ model_text.pt
│       ├─ model_image.pt
│       ├─ model_fusion.pt
│       └─ FPN_fusion.pt
│
├─ main.py                 # Main program
└─ README.md               # Project description
```

## Environment Setup

It is recommended to use Python 3.10 or higher, and install the following dependencies:

```bash
# Optional: create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install pillow
pip install scikit-learn
pip install numpy
pip install tokenizers
```

> Note: If you have a GPU, please select the appropriate PyTorch version according to your CUDA version.

## Data Preparation

1. **Image Data**  
   Place the original images in `data/train_L/` with file names formatted as `{image_ID}.png`.

2. **Annotation File**  
   Place the JSON file `multi_data_0.05.json` in the `data/` folder, with the format:

```json
[
    {
        "image_ID": 1,
        "description": "text description",
        "answers": ["Heat pipe"]
    },
    ...
]
```

3. **Data Splitting**  
   The program will automatically split the dataset into training and validation sets, generating:
   - `train_data.json` / `text_data.json`
   - Corresponding image folders: `train_data/` / `text_data/`

## Running Steps

1. **Generate Tokenizer**

```python
from main import word_processor
processor = word_processor()
```

The tokenizer will be saved to `data/model/tokenizer.json` and used for text encoding.

2. **Create Dataset**

```python
from main import TextImageDataset, Dataset_split

base_path = 'data/'
image_file1, json_file1, image_file2, json_file2 = Dataset_split(base_path)
train_set = TextImageDataset(image_file1, json_file1, processor, r1=1)
val_set   = TextImageDataset(image_file2, json_file2, processor, r1=2)
```

3. **Train and Validate Models**  
   - Text feature network: `TransformerModel`  
   - Image feature network: `fastvit_t8`  
   - Fusion network: `FullModel`  
   - Multi-source feature judgment network: `BiFPNNet`  

The full training pipeline is implemented in `main.py`. You can run it directly:

```bash
python main.py
```

4. **Results Saving**  
Models and validation results are automatically saved in `result/CASE1/`:

- `model_text.pt` — Text network
- `model_image.pt` — Image network
- `model_fusion.pt` — Fusion network
- `FPN_fusion.pt` — Multi-source judgment network
- Corresponding validation results `.txt` files

## Notes

1. Ensure GPU availability; training on CPU may be slow.  
2. The default paths in `main.py` use absolute paths. Please adjust `base_path` and `result` paths according to your environment.  
3. You can modify the following parameters to control training:
   - `page`: experiment number
   - `device`: training device
   - `num_epochs`: number of epochs
   - `batch_size`: batch size

## Contact

If you have questions, please open a GitHub Issue or contact the project maintainer by email.
