# Cultural Item Classification

## Project Overview
This project tackles the task of classifying cultural items into three taxonomic classes: **Cultural Agnostic (C.A)**, **Cultural Representative (C.R)**, and **Cultural Exclusive (C.E)**. The approach compares the strengths and limitations of a large language model-based classifier and a knowledge-based, non-LM method exploiting structured data from Wikidata.

## Repository Structure
- `data/`
  - `test_unlabeled.csv`: The test set provided for prediction.
- `notebooks/`
  - `model1.ipynb`: Code for training and evaluating the LM-based classifiers (DistilBERT and RoBERTa).
  - `model2.ipynb`: Code for the Non-LM based classifiers (Logistic Regression, Random Forest, SVM) and final predictions.
- `Report.pdf`: Comprehensive report discussing the methodology, experiments, and classification metrics.

## Requirements and Setup
**⚠️ Important:** To run the notebooks successfully, you must provide your own **Hugging Face Token**.
- The dataset (`sapienzanlp/nlp2025_hw1_cultural_dataset`) and the pre-trained models are fetched directly from the Hugging Face Hub.
- Before executing the code, please insert your personal token in the notebooks where requested:
  ```python
  from huggingface_hub import login
  login(token="YOUR_HF_TOKEN")

## Dataset
The models are trained and evaluated on the sapienzanlp/nlp2025_hw1_cultural_dataset. The inputs include item metadata such as name, description, type, and category. Testing is performed on the unlabeled dataset test_unlabeled.csv.

## Methodology
### 1. LM-Based Classifier
- **Models Used**: distilbert-base-uncased and roberta-base.

- **Approach**: The text fields were concatenated and tokenized. The first 3 layers of the Transformer encoder were frozen to prevent catastrophic forgetting and reduce overfitting.

- **Selection**: DistilBERT outperformed RoBERTa on validation accuracy and was selected as the primary LM-based classifier.

### 2. Non-LM Based Classifier
- **Models Used**: Logistic Regression, Random Forest, and Support Vector Machine (SVM).

- **Features Extracted**:

  - **TF-IDF**: Max 5000 features from the concatenated text.

  - **Word2Vec**: Word embeddings of size 50 to capture semantic relationships.

  - **Wikidata**: Country-related metadata properties extracted via API (and cached).

- **Imbalance Handling**: The Synthetic Minority Over-sampling Technique (SMOTE) was used to address class imbalance.

- **Selection**: SVM with a linear kernel yielded the best validation accuracy among the traditional models.

## Results
LM-Based (DistilBERT): Accuracy: 0.770 | Macro F1: 0.752

Non-LM Based (SVM): Accuracy: 0.660 | Macro F1: 0.640

Note: Both methods performed best on the "Cultural Agnostic" class. They exhibited lower recall for the "Cultural Exclusive" class, primarily due to its sparsity and linguistic ambiguity.
