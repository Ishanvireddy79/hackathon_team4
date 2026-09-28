# AspectSense: Aspect-Based Sentiment Analysis

## Task
Predict the sentiment (Positive, Negative or Neutral) of a given aspect term inside a review. One review can contain different sentiments for different aspects, so a single score per review fails.
Official metric: Macro F1. Secondary: accuracy.

## Data
- 8,000 train rows and 2,000 test rows
- Train labels: Neutral 3,603, Positive 2,431, Negative 1,966
- 4,373 unique reviews; 2,330 of them contain aspects with different sentiments
- Validation split: stratified 10% (7,200 train, 800 validation)

## Results (validation)
- Best Macro F1: 0.844
- Best accuracy: 0.848
- Baseline, TF-IDF + logistic regression, review only: 0.377 Macro F1
- Baseline, TF-IDF + logistic regression, aspect term plus nearby words: 0.670 Macro F1
- DeBERTa-v3-base, aspect and review pair: 0.844 Macro F1

## Training by epoch
| Epoch | Val loss | Macro F1 | Accuracy |
|---|---|---|---|
| 1 | 0.516 | 0.793 | 0.795 |
| 2 | 0.532 | 0.824 | 0.830 |
| 3 (best) | 0.604 | 0.844 | 0.848 |
| 4 | 0.708 | 0.842 | 0.846 |

Validation loss rose after epoch 1 (overfitting), so the best checkpoint was chosen by Macro F1, not the last epoch.

## Per-class results
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.8729 | 0.8020 | 0.8360 | 197 |
| Neutral | 0.8382 | 0.8778 | 0.8575 | 360 |
| Positive | 0.8388 | 0.8354 | 0.8371 | 243 |

Macro F1 0.8435, accuracy 0.8462.

## Confusion matrix
Rows are true labels, columns are predicted. Order: Negative, Neutral, Positive.
[[158, 29, 10],
 [15, 316, 29],
 [8, 32, 203]]

Error analysis: 123 errors in total. About 85% involve Neutral. Direct Negative/Positive flips are rare (10 + 8 = 18 of 800). Weakest class: Negative (F1 0.836), but all three classes are within about 2 points of each other.

## Test predictions
Neutral 917, Positive 586, Negative 497 (2,000 rows), close to the training class mix.

## How it works
1. Explore the data and label balance.
2. Clean a data trap: the aspect term "nan" was read as missing, fixed with keep_default_na=False.
3. Map labels: Negative 0, Neutral 1, Positive 2.
4. Input as a pair: [CLS] aspect [SEP] review [SEP], truncating only the review, max length 128, dynamic padding.
5. Fine-tune microsoft/deberta-v3-base with a new 3-class head.
6. Weighted cross-entropy with inverse class frequency weights: Negative 1.357, Neutral 0.740, Positive 1.097.
7. Select the best checkpoint by Macro F1.
8. Predict the 2,000 test rows, write submission.csv, and run the organizers' validator.

## Setup
- Google Colab, T4 GPU, Python 3
- PyTorch, Hugging Face Transformers (Trainer), pandas, datasets, scikit-learn
- Learning rate 2e-5, batch size 16, 4 epochs, seed 42, max length 128
- AdamW, weight decay 0.01, 180 warmup steps (10% of 1,800), linear decay, fp16
- No API keys and no external inference API

## Problems fixed
- warmup_ratio was rejected by the newer Transformers version, so we used warmup_steps=180
- tokenizer= was renamed, so we used processing_class=
- FP16 unscale error, fixed with model = model.float()
- "nan" aspect read as missing, fixed with keep_default_na=False

## Limitations and not tried
- Neutral is the ambiguous boundary class where most errors occur
- Not tried: the context_with_aspect_tag variant, multiple seeds, cross-validation, ensembles, larger models
- The website does not run the model; it only presents the results