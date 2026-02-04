# SleepFM Codebase Summary and Feature Analysis

## 📋 Executive Summary

**SleepFM** (Sleep Foundation Model) is a multi-modal deep learning framework for sleep analysis using polysomnography (PSG) data. It was accepted at **ICML 2024** and represents the first multi-modal foundation model specifically designed for comprehensive sleep analysis.

The codebase provides a complete pipeline for:
1. Preprocessing PSG data from EDF files
2. Creating train/validation/test splits
3. Pretraining using contrastive learning across multiple modalities
4. Generating embeddings for downstream tasks
5. Evaluating sleep stage classification performance

---

## 🏗️ Repository Structure

```
sleepfm-codebase/
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── sleepfm/                            # Main source code
    ├── config.py                       # Configuration and paths
    ├── utils.py                        # Utility functions
    ├── 0_extract_pretraining_data.py   # Step 0: Data extraction
    ├── 1_prepare_dataset.py            # Step 1: Dataset preparation
    ├── 2_pretrain.py                   # Step 2: Model pretraining
    ├── 3_generate_embed_pretraining.py # Step 3: Embedding generation
    ├── 4_classification_eval_pretraining.py # Step 4: Evaluation
    ├── model/                          # Model definitions
    │   ├── models.py                   # Neural network architectures
    │   └── dataset.py                  # PyTorch dataset classes
    ├── checkpoint/                     # Pre-trained model checkpoints
    │   └── best.pt                     # Best model weights
    └── bash_scripts/                   # Execution scripts
        ├── 0_extract_pretraining_data.sh
        ├── 1_prepare_dataset.sh
        ├── 2_pretrain.sh
        ├── 3_generate_embed_pretraining.sh
        └── 4_classification_eval_pretraining.sh
```

---

## 🔬 Technical Components

### 1. Data Modalities

SleepFM processes **three physiological signal modalities**:

| Modality | Channels | Description |
|----------|----------|-------------|
| **Respiratory** | CHEST, SaO2, ABD | Chest movement, blood oxygen saturation, abdominal movement |
| **Sleep Stages (Brain Activity)** | C3-M2, C4-M1, O1-M2, O2-M1, E1-M2 | EEG channels for sleep staging |
| **EKG (Cardiac)** | ECG | Electrocardiogram signal |

**Total Channels Supported**: 13 channels from PSG recordings
- F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, E1-M2, Chin1-Chin2, ABD, CHEST, AIRFLOW, SaO2, ECG

### 2. Sleep Stage Classification

The model classifies **5 sleep stages**:

| Stage | Label | Description |
|-------|-------|-------------|
| 0 | Wake | Wakefulness |
| 1 | Stage 1 (N1) | Light sleep |
| 2 | Stage 2 (N2) | Light sleep |
| 3 | Stage 3 (N3) | Deep sleep |
| 4 | REM | Rapid eye movement sleep |

### 3. Neural Network Architecture

#### EffNet (1D EfficientNet-based)

The core model is a **1D Convolutional Neural Network** based on EfficientNet principles:

```
Architecture Components:
├── Input Layer: Conv1d → BatchNorm
├── Stage 2-8: MBConv blocks with increasing depth
│   └── MBConv (Mobile Inverted Bottleneck Convolution)
│       ├── Bottleneck layers with expansion
│       ├── Depthwise separable convolutions
│       └── Residual connections (when stride=1)
├── Stage 9: Conv1d (1x1 convolution)
├── Adaptive Average Pooling
├── Dropout
└── Fully Connected Layer → 512-dimensional embeddings
```

**Key Hyperparameters**:
- **Depth Configuration**: [1, 2, 2, 3, 3, 3, 3]
- **Channel Configuration**: [32, 16, 24, 40, 80, 112, 192, 320, 1280]
- **Expansion Factor**: 6
- **Embedding Dimension**: 512

#### Model Variants

1. **EffNet** - For contrastive learning (outputs 512-dim embeddings)
2. **EffNetSupervised** - For direct classification (outputs class probabilities)

---

## 🔄 Pipeline Scripts

### Step 0: Data Extraction (`0_extract_pretraining_data.py`)

**Purpose**: Extract 30-second epochs from raw PSG data

**Key Features**:
- Processes EDF (European Data Format) files
- Extracts signals and arousal labels from MATLAB files
- Resamples data to target sampling rate (256 Hz default)
- Supports parallel processing with configurable threads
- Saves epoch data as NumPy arrays (.npy files)

**Arguments**:
```python
--data_path         # Path to raw EDF files
--save_path         # Output path for processed data
--num_files         # Number of files to process (-1 for all)
--chunk_duration    # Epoch duration in seconds (default: 30.0)
--num_threads       # Parallel processing threads (default: 4)
--target_sampling_rate  # Target sample rate (default: 256)
```

### Step 1: Dataset Preparation (`1_prepare_dataset.py`)

**Purpose**: Create train/validation/test splits and organize data

**Key Features**:
- Patient-level splitting (no data leakage)
- Creates pretrain/train/valid/test splits
- Handles label mapping and event organization
- Supports sampling for debugging
- Outputs pickle files for efficient loading

**Split Strategy**:
- Pretrain: 75% of subjects
- Train: Subset of remaining 25%
- Valid: 10% of train subjects
- Test: Configurable size (default: 100 subjects)

### Step 2: Pretraining (`2_pretrain.py`)

**Purpose**: Train the multi-modal contrastive learning model

**Key Features**:
- Two contrastive learning modes:
  1. **Pairwise**: Standard pairwise contrastive loss between modalities
  2. **Leave-One-Out**: Novel approach comparing one modality to average of others
- Learnable temperature parameter
- Multi-GPU support via DataParallel
- Checkpoint saving and resumption
- Detailed logging with loss and accuracy metrics

**Arguments**:
```python
--dataset_dir       # Data directory
--dataset_file      # Dataset pickle file
--batch_size        # Training batch size (default: 16)
--lr                # Learning rate (default: 1e-4)
--epochs            # Number of training epochs (default: 100)
--mode              # "pairwise" or "leave_one_out"
--modality_types    # Comma-separated list of modalities
```

### Step 3: Embedding Generation (`3_generate_embed_pretraining.py`)

**Purpose**: Generate embeddings from pretrained models for evaluation

**Key Features**:
- Loads trained model checkpoints
- Generates normalized embeddings for all three modalities
- Processes train/valid/test splits
- Saves embeddings as pickle files for downstream tasks

### Step 4: Classification Evaluation (`4_classification_eval_pretraining.py`)

**Purpose**: Evaluate sleep stage classification using learned embeddings

**Key Features**:
- Trains logistic regression or XGBoost on embeddings
- Supports individual modalities or combined embeddings
- Calculates comprehensive metrics:
  - Accuracy
  - AUROC (per-class and macro/weighted)
  - AUPRC (per-class and macro/weighted)
  - Confusion matrix
- Bootstrap confidence intervals
- Saves models, probabilities, and reports

---

## 📦 Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.0.1 | Deep learning framework |
| NumPy | 1.25.2 | Numerical computing |
| pandas | 2.1.1 | Data manipulation |
| scikit-learn | 1.3.1 | Machine learning utilities |
| MNE | 1.5.1 | EEG/MEG data processing |
| pyedflib | 0.1.37 | EDF file reading |

### Additional Libraries

| Library | Purpose |
|---------|---------|
| loguru | Logging |
| tqdm | Progress bars |
| matplotlib/seaborn | Visualization |
| wandb | Experiment tracking |
| XGBoost | Gradient boosting classifier |
| einops | Tensor operations |
| transformers | Hugging Face utilities |

---

## 🎯 Key Features & Innovations

### 1. Leave-One-Out Contrastive Learning

A novel contribution where instead of computing pairwise losses between modalities, each modality is compared against the average embedding of all other modalities. This approach:
- Captures holistic multi-modal relationships
- Outperforms standard pairwise contrastive learning
- Better suited for 3+ modality scenarios

### 2. Multi-Modal Foundation Model

First foundation model specifically designed for sleep analysis that:
- Processes respiratory, brain activity, and cardiac signals jointly
- Learns shared representations across modalities
- Enables cross-modal retrieval (48% top-1 accuracy from 90,000 candidates)

### 3. Transfer Learning Capabilities

The learned embeddings:
- Outperform end-to-end CNNs with simple linear classifiers
- Sleep stage classification: macro AUROC 0.88 vs 0.72
- Sleep disordered breathing detection: AUROC 0.85 vs 0.69

### 4. Flexible Architecture

- Modular design allowing different modality combinations
- Configurable network depth and width
- Support for both contrastive pretraining and supervised learning

---

## 🗂️ Data Flow

```
Raw PSG Data (EDF files)
         │
         ▼
┌─────────────────────────────────┐
│ 0_extract_pretraining_data.py  │
│ - Extract channels              │
│ - Create 30-second epochs       │
│ - Resample to 256 Hz            │
└─────────────────────────────────┘
         │
         ▼
    NumPy Arrays (.npy)
    + Label Pickle Files
         │
         ▼
┌─────────────────────────────────┐
│    1_prepare_dataset.py        │
│ - Create patient-level splits   │
│ - Organize by event/label       │
└─────────────────────────────────┘
         │
         ▼
    Dataset Pickle Files
         │
         ▼
┌─────────────────────────────────┐
│       2_pretrain.py            │
│ - Contrastive learning          │
│ - Multi-modal training          │
└─────────────────────────────────┘
         │
         ▼
    Model Checkpoints (.pt)
         │
         ▼
┌─────────────────────────────────┐
│ 3_generate_embed_pretraining.py│
│ - Generate embeddings           │
│ - 512-dim per modality          │
└─────────────────────────────────┘
         │
         ▼
    Embedding Pickle Files
         │
         ▼
┌─────────────────────────────────┐
│4_classification_eval_pretraining│
│ - Train linear classifiers      │
│ - Evaluate performance          │
└─────────────────────────────────┘
         │
         ▼
    Performance Metrics
    (AUROC, AUPRC, Accuracy)
```

---

## 🔧 Configuration (`config.py`)

### Path Settings
```python
PATH_TO_RAW_DATA = "/path/to/raw/data/"
PATH_TO_PROCESSED_DATA = "/path/to/processed/data/"
```

### Label Mappings
- Supports multiple label formats (Sleep stage W, N1, N2, N3, R, etc.)
- Unified internal representation (Wake, Stage 1-3, REM)

### Channel Organization
- Respiratory: 3 channels (CHEST, SaO2, ABD)
- Sleep Stages: 5 channels (EEG)
- EKG: 1 channel

---

## 📊 Dataset Classes (`model/dataset.py`)

### EventDataset
For contrastive learning - returns data tuples for all modalities

### EventDatasetSupervised
For supervised learning - returns (data, label) pairs for single modality

Both support:
- Pickle file loading
- Split selection (pretrain/train/valid/test)
- Modality selection
- Combined modality option

---

## 🚀 Usage Example

```bash
# Step 0: Extract data
python 0_extract_pretraining_data.py \
    --data_path /path/to/raw/data \
    --save_path /path/to/save \
    --num_files -1 \
    --chunk_duration 30.0

# Step 1: Prepare dataset
python 1_prepare_dataset.py \
    --dataset_dir /path/to/processed/data

# Step 2: Pretrain model
python 2_pretrain.py \
    --epochs 100 \
    --mode leave_one_out \
    --batch_size 32 \
    --lr 1e-3

# Step 3: Generate embeddings
python 3_generate_embed_pretraining.py /path/to/output \
    --splits train,valid,test

# Step 4: Evaluate
python 4_classification_eval_pretraining.py \
    --output_file /path/to/output \
    --modality_type combined
```

---

## 📈 Performance Metrics (from Paper)

| Task | Model | AUROC | AUPRC |
|------|-------|-------|-------|
| Sleep Stage Classification | SleepFM (LR) | **0.88** | **0.72** |
| Sleep Stage Classification | End-to-end CNN | 0.72 | 0.48 |
| Sleep Disordered Breathing | SleepFM (LR) | **0.85** | **0.77** |
| Sleep Disordered Breathing | End-to-end CNN | 0.69 | 0.61 |

---

## 📝 Utility Functions (`utils.py`)

### Data I/O
- `save_data()` / `load_data()`: Save/load pickle or JSON files
- `getEDFFilenames()`: Get list of EDF files
- `getChannelLabels()`: Extract channel labels from EDF
- `read_events_file_as_df()`: Parse event files

### Model Training & Evaluation
- `train_model()`: Train logistic regression or XGBoost
- Computes: accuracy, classification report, confusion matrix
- Bootstrap confidence intervals for AUROC/AUPRC

### Data Processing
- `filter_edf_events_file_pair()`: Filter valid EDF/event pairs
- `get_all_edf_and_events_file_pair()`: Match EDF with event files

---

## 🎓 Citation

```bibtex
@inproceedings{thapa2024sleepfm,
  title={SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals},
  author={Rahul Thapa and Bryan He and Magnus Ruud Kjaer and Hyatt Moore and Gauri Ganjoo and Emmanuel Mignot and James Zou},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🔮 Future Directions (from README)

The authors are working on:
- Larger model versions
- Architectural improvements
- Training on more data
- Updated codebase and model releases
