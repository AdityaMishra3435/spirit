# SPIRIT: Short-term Prediction of Solar IRradiance for Zero-Shot Transfer Learning


[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.10307-b31b1b.svg)](https://arxiv.org/abs/2502.10307)

## 📌 Overview
SPIRIT is a novel machine learning system for **solar irradiance forecasting**, leveraging **foundation models** and **zero-shot transfer learning** to enable accurate predictions even for **new locations without historical data**. This system significantly outperforms traditional models in both **nowcasting** (real-time forecasting) and **short-term forecasting** (1 to 4-hour predictions). SPIRIT implements a hybrid architecture with a PyTorch-based transformer model for forecasting and an XGBoost-based model for nowcasting, combining the strengths of both frameworks.


### Key Features:
- **Foundation model-powered**: Uses pre-trained vision transformers (ViT) for feature extraction
- **Physics-inspired features**: Incorporates clear sky models and solar geometry data
- **Hyperparameter optimization**: Uses Optuna for model tuning
- **Zero-shot learning**: Works at new locations without training data
  
## 📁 Repository Structure
```plaintext
📂 spirit
 ├── 📁 models         # Model implementations
 │   ├── nowcast.py            
 │   └── forecast.py           
 ├── 📁 data           # Data processing
 │   ├── config.py             
 │   ├── main.py               
 │   ├── README.md             
 │   ├── modules                    
 │   │   ├── data_creation.py  
 │   │   ├── download.py       
 │   │   ├── embedding_generation.py 
 │   │   ├── extraction.py     
 │   │   └── preprocessing.py  
 │   └── utils                     
 │       ├── data_utils.py     
 │       ├── file_utils.py     
 │       └── __init__.py       
 └── 📜 README.md                 
```

## 🔧 Implementation

### Nowcasting Model 
The implementation of the nowcasting system uses XGBoost to predict current solar irradiance from sky images:
- **Input**: Single sky image processed through a Vision Transformer and auxiliary features
- **Architecture**: Gradient-boosted decision trees (XGBoost) with tree pruning, feature subsampling, and optimized hyperparameters for non-linear regression 
- **Output**: Global Horizontal Irradiance prediction

### Forecasting Model
The implementation of the forecasting system uses a transformer-based architecture for time series predictions:
- **Input**: Sequence of sky images and the corresponding auxiliary data
- **Architecture**: Transformer encoder with residual MLP blocks, multi-head attention mechanism, and temporal position embeddings
- **Output**: GHI predictions for future time steps (1-4 hours)

## ⚡ Installation
Ensure you have Python 3.8+ installed. Then, clone the repository and install dependencies:
```bash
git clone https://github.com/AdityaMishra3435/spirit.git
cd spirit
pip install -r requirements.txt
```

## 🚀 Usage

**Nowcasting**
```sh
cd models  
python nowcast.py  
```
**Forecasting**
```sh
cd models  
python forecast.py  
```
**Performance Metrics**  
SPIRIT is evaluated using the metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Normalized Mean Absolute Percentage Error (nMAP) and Coefficient of Determination (R2 Score) 

## 📜 Citation
If you use SPIRIT in your research, please cite the paper:
```bibtex
@article{spirit2025,
  title={SPIRIT: Short-term Prediction of Solar Irradiance for Zero-Shot Transfer Learning Using Foundation Models},
  author={Aditya Mishra, T Ravindra, Srinivasan Iyengar, Shivkumar Kalyanaraman, Ponnurangam Kumaraguru},
  year={2025}
}
```

## 👥 Contributors
- **Aditya Mishra** (IIIT Hyderabad)
- **T Ravindra** (IIIT Hyderabad)
- **Srinivasan Iyengar** (Microsoft)
- **Shivkumar Kalyanaraman** (Microsoft)
- **Ponnurangam Kumaraguru** (IIIT Hyderabad)
