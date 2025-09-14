# Punctuation Restoration for Mental Health Conversations

<p align="center">
  <img src="C:\Users\LENOVO\OneDrive\Desktop\NLP_AUGNITO\punctuation_restoration\images\project_banner.png" alt="Project Banner" />
</p>

A BERT-based neural network system for automatic punctuation restoration in mental health conversation datasets. This project addresses the critical need for accurate punctuation in therapeutic dialogue systems, achieving 84% F1-score through advanced deep learning techniques.

A BERT-based neural network system for automatic punctuation restoration in mental health conversation datasets. This project addresses the critical need for accurate punctuation in therapeutic dialogue systems, achieving 84% F1-score through advanced deep learning techniques.

## 🎯 Project Overview

This system automatically restores punctuation marks (periods, commas, question marks, exclamation marks, colons, semicolons) in unpunctuated text from mental health conversations. The model is specifically fine-tuned for the therapeutic domain to handle sensitive and context-aware dialogue.

### Key Features

- **BERT-based Architecture**: Leverages pre-trained transformer models for contextual understanding
- **Domain-Specific Fine-tuning**: Optimized for mental health conversation patterns
- **Class Imbalance Handling**: Weighted cross-entropy loss for handling rare punctuation marks
- **High Performance**: Achieves 84% F1-score with robust overfitting detection
- **Production Ready**: Complete inference pipeline with text post-processing

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **F1 Score** | 84.0% |
| **Validation Loss** | 0.067 |
| **Training Epochs** | 9 (optimal) |
| **Dataset Size** | 2,471 cleaned samples |

### Class-wise Performance

| Punctuation | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Period (.) | 0.89 | 0.92 | 0.90 |
| Comma (,) | 0.85 | 0.87 | 0.86 |
| Question (?) | 0.78 | 0.74 | 0.76 |
| Exclamation (!) | 0.71 | 0.68 | 0.69 |
| Colon (:) | 0.65 | 0.61 | 0.63 |
| Semicolon (;) | 0.58 | 0.52 | 0.55 |

## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
CUDA (optional, for GPU training)
```

### Dependencies

```bash
pip install torch transformers pandas scikit-learn tqdm pyyaml numpy
```

### Clone Repository

```bash
git clone [https://github.com/LaxmiNarayanSharma00/Punctuation_Restoration]
```

## 📁 Project Structure

```
punctuation-restoration/
├── src/
│   ├── model.py           # BERT-based model architecture
│   ├── dataset.py         # Dataset preprocessing and loading
│   ├── trainer.py         # Training loop with validation
│   ├── inference.py       # Punctuation restoration inference
│   └── utils.py           # Utility functions and helpers
├── data/
│   └── mental_health_dataset.csv
├── checkpoints/           # Model checkpoints directory
├── main.py               # Main training and inference script
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Training

Train the model on your mental health conversation dataset:

```bash
python main.py train --data_path data/mental_health_dataset2.csv --checkpoint_dir checkpoints/ --epochs 6
```

#### Training Parameters

- `--data_path`: Path to CSV file with 'Response' column
- `--checkpoint_dir`: Directory to save model checkpoints
- `--epochs`: Number of training epochs (default: 7)
- `--batch_size`: Training batch size (default: 16)
- `--lr`: Learning rate (default: 3e-5)
- `--max_len`: Maximum sequence length (default: 128)
- `--resume_from`: Resume training from checkpoint (optional)

### 2. Inference

Restore punctuation in unpunctuated text:

```bash
python main.py inference --model_path checkpoints/best_model.pt --text "yesterday i met my friend at the park we talked about our college days laughed a lot and even planned a trip for next month can you believe it we havent seen each other for five years time really flies doesnt it anyway we decided to visit goa in december i cant wait"
```

**Output:**
```

Yesterday, I met my friend at the park. We talked about our college days, laughed a lot, and even planned a trip for next month. Can you believe it? We haven't seen each other for five years. Time really flies, doesn't it? Anyway, we decided to visit Goa in December. I can't wait!

```

### 3. Programmatic Usage

```python
from src.inference import PunctuationRestorer

# Load trained model
restorer = PunctuationRestorer(model_path="checkpoints/best_model.pt")

# Restore punctuation
unpunctuated_text = "hello how are you feeling today"
result = restorer.restore_punctuation(unpunctuated_text)
print(result)  # "Hello, how are you feeling today?"
```

## 🔧 Model Architecture

### BERT-based Punctuation Restoration

```python
class BertForPunctuation(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', 
                 num_punct_classes=7, dropout=0.3):
        super(BertForPunctuation, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_punct_classes)
```

### Key Components

1. **BERT Encoder**: 12-layer transformer with 768-dimensional hidden states
2. **Dropout Layer**: 0.3 dropout rate for regularization
3. **Linear Classifier**: Maps 768 features to 7 punctuation classes
4. **Weighted Loss**: Handles severe class imbalance in punctuation distribution

### Punctuation Classes

| Class ID | Punctuation | Description |
|----------|-------------|-------------|
| 0 | `O` | No punctuation |
| 1 | `,` | Comma |
| 2 | `.` | Period |
| 3 | `?` | Question mark |
| 4 | `!` | Exclamation mark |
| 5 | `:` | Colon |
| 6 | `;` | Semicolon |

## 📈 Training Details

### Dataset Statistics

- **Original Records**: 5,000
- **After Cleaning**: 2,471 samples
- **Duplicates Removed**: 1,040
- **Null Values**: 4
- **Mean Token Length**: 175.51 tokens

### Weighted Cross-Entropy Loss

The model uses weighted cross-entropy loss to handle severe class imbalance:

```
Weight_i = Total_Samples / (Num_Classes × Class_i_Samples)
```

### Class Weights Applied

| Punctuation | Count | Weight |
|-------------|-------|--------|
| Period (.) | 23,757 | 0.15 |
| Comma (,) | 17,245 | 0.20 |
| Question (?) | 2,091 | 1.67 |
| Exclamation (!) | 959 | 3.64 |
| Colon (:) | 764 | 4.57 |
| Semicolon (;) | 415 | 8.41 |

### Training Configuration

- **Optimizer**: AdamW with β₁=0.9, β₂=0.999
- **Learning Rate**: 3×10⁻⁵ with linear warmup
- **Batch Size**: 16
- **Max Sequence Length**: 128 tokens
- **Early Stopping**: Patience of 3 epochs
- **Regularization**: 0.3 dropout + weight decay

## 🔄 Advanced Usage

### Resume Training

```bash
python main.py train \
    --data_path data/mental_health_dataset.csv \
    --resume_from checkpoints/checkpoint_epoch6.pt \
    --epochs 15
```

### Custom Dataset Format

Your CSV file should contain a 'Response' column with conversational text:

```csv
Response
"Hello, how are you feeling today?"
"I've been struggling with anxiety lately."
"What coping strategies have you tried?"
```

### Batch Inference

```python
from src.inference import PunctuationRestorer

restorer = PunctuationRestorer("checkpoints/best_model.pt")

texts = [
    "i feel anxious about the meeting tomorrow",
    "can you help me understand my emotions",
    "what should i do when i feel overwhelmed"
]

for text in texts:
    restored = restorer.restore_punctuation(text)
    print(f"Original: {text}")
    print(f"Restored: {restored}\n")
```

## 📝 Data Preprocessing

The system automatically handles:

1. **Text Normalization**: Lowercase conversion, special character handling
2. **Tokenization**: BERT WordPiece tokenization
3. **Label Generation**: Synthetic unpunctuated text creation
4. **Sequence Alignment**: Token-level punctuation label alignment
5. **Padding/Truncation**: Uniform sequence length processing

## 🎯 Use Cases

### Clinical Applications

- **Therapeutic Chatbots**: Real-time punctuation restoration
- **Clinical Note Processing**: Automated transcription post-processing
- **Research Data Preparation**: Standardizing conversation datasets
- **Quality Assurance**: Improving readability of therapy transcripts

### Technical Integration

- **API Deployment**: REST API for punctuation restoration service
- **Batch Processing**: Large-scale text processing pipelines
- **Real-time Applications**: Live chat punctuation enhancement
- **NLP Preprocessing**: Preparing text for downstream tasks

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python main.py train --batch_size 8
   ```

2. **Training Instability**
   ```bash
   # Lower learning rate
   python main.py train --lr 1e-5
   ```

3. **Poor Performance on Rare Punctuation**
   - Increase class weights for minority classes
   - Use data augmentation techniques
   - Collect more samples with rare punctuation

### Performance Optimization

- **GPU Training**: Use CUDA-enabled PyTorch for faster training
- **Mixed Precision**: Implement automatic mixed precision (AMP)
- **Gradient Accumulation**: Simulate larger batch sizes
- **Model Quantization**: Reduce model size for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 code style
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Vaswani, A., et al. (2017). Attention Is All You Need.
3. Mental Health Conversation Dataset Processing Techniques
4. Weighted Cross-Entropy for Imbalanced Classification

## 🙏 Acknowledgments

- **Hugging Face Transformers**: For providing BERT implementations
- **Mental Health Research Community**: For dataset contributions
- **PyTorch Team**: For the deep learning framework
- **Clinical NLP Community**: For domain expertise and validation

---

**Built with ❤️ for better mental health conversation processing**
