# AspectSense — Aspect-Based Sentiment Analysis

> Aspect-conditioned sentiment classification with a fine-tuned
> DeBERTa-v3 encoder: given a review and an aspect term, predict the
> sentiment (Positive, Negative or Neutral) toward that specific aspect,
> not toward the review as a whole.

**🌐 [Project website](YOUR_NETLIFY_LINK)** ·
**📓 [Notebook](./ABSA_starter_notebook.ipynb)** ·
**📝 [Project notes](./notes.md)** ·
**✅ [Verified submission](./submission_verified.csv)**

---

## What this is

A deep-learning solution for Challenge 01 of the PDNC Hackathon
(Department of AI&DS, NLP & Deep Neural Networks). It does three things
a single review-level classifier can't:

1. **Conditions on the aspect.** The aspect term is fed to the model
   together with the review as a sentence pair, so attention links the
   aspect to its own opinion words. One review can score Positive for
   one aspect and Negative for another.
2. **Optimizes the official metric.** Training uses class-weighted loss,
   and the best checkpoint is chosen by Macro F1, not accuracy or the
   last epoch, so the large Neutral class can't win by default.
3. **Reports its own weak spots.** An error analysis shows where the
   model fails: about 85% of its mistakes involve Neutral, and direct
   Positive/Negative flips are rare.

A live Gradio demo (run from the notebook) lets you type a review and an
aspect and see the predicted sentiment.

---

## Results (validation set)

| Metric                        | Value  | Notes                                       |
|-------------------------------|--------|---------------------------------------------|
| Macro F1 (official metric)    | **0.8435** | stratified 10% validation split, n=800  |
| Accuracy                      | 0.8462 | secondary metric                            |
| Best baseline Macro F1        | 0.670  | TF-IDF + logistic regression, aspect and nearby words |
| Review-only baseline Macro F1 | 0.377  | TF-IDF + logistic regression, no aspect     |
| Negative F1                   | 0.8360 | weakest class, recall 0.8020                |
| Neutral F1                    | 0.8575 | strongest class, recall 0.8778              |
| Positive F1                   | 0.8371 | recall 0.8354                               |
| Errors involving Neutral      | ~85%   | 105 of 123 errors                           |

Validation loss rose after epoch 1 while Macro F1 peaked at epoch 3, a
sign of overfitting. That is why the best checkpoint is kept by Macro F1
instead of using the final epoch.

Confusion matrix (rows = true, columns = predicted; order Negative,
Neutral, Positive):

```
[[158  29  10]
 [ 15 316  29]
 [  8  32 203]]
```

Test predictions (2,000 rows): Neutral 917, Positive 586, Negative 497,
close to the training class mix.

---

## Stack

| Layer          | Tools                                                  |
|----------------|--------------------------------------------------------|
| Compute        | Google Colab, T4 GPU, Python 3                         |
| Deep learning  | PyTorch, Hugging Face Transformers (`Trainer`)         |
| Model          | `microsoft/deberta-v3-base` with a new 3-class head    |
| Data           | pandas, Hugging Face `datasets`, scikit-learn          |
| Tokenization   | sentencepiece, tiktoken                                |
| Baselines      | TF-IDF + logistic regression (scikit-learn)            |
| Demo           | Gradio                                                 |
| Website        | Vanilla HTML/CSS/JS (no framework)                     |

No API keys were used and no external inference API was called. All
training and prediction ran on Colab.

---

## Method

1. **Explore.** 8,000 train and 2,000 test rows. Train labels: Neutral
   3,603, Positive 2,431, Negative 1,966. Of 4,373 unique reviews, 2,330
   contain aspects with different sentiments.
2. **Clean.** The aspect term "nan" was read by pandas as missing, so the
   CSVs load with `keep_default_na=False`.
3. **Pair the input.** `[CLS] aspect [SEP] review [SEP]`, truncating only
   the review, max length 128, dynamic padding.
4. **Fine-tune** DeBERTa-v3-base: learning rate 2e-5, batch size 16,
   4 epochs, AdamW, weight decay 0.01, 180 warmup steps then linear
   decay, fp16, seed 42.
5. **Weight the loss.** Cross-entropy with inverse class frequency
   weights: Negative 1.357, Neutral 0.740, Positive 1.097.
6. **Select and predict.** Keep the best checkpoint by Macro F1, predict
   the 2,000 test rows, write `submission.csv`, then run the organizers'
   validator to produce `submission_verified.csv`.

---

## Run the demo locally

Requirements: a Colab T4 GPU (or any machine with Python 3.11+;
inference also runs on CPU, more slowly).

```bash
# 1. Install dependencies
pip install torch transformers datasets scikit-learn pandas sentencepiece tiktoken accelerate gradio

# 2. Load the saved model (after training and saving it as absa_best)
python - <<'EOF'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("absa_best")
model = AutoModelForSequenceClassification.from_pretrained("absa_best").eval()
print(model.config.id2label)
EOF
```

The Gradio demo cell at the end of the notebook builds the interface and
prints a public link when run with `share=True`.

---

## Reproducing the experiments

Open `ABSA_starter_notebook.ipynb` in Google Colab, switch the runtime to
a T4 GPU, upload the organizers' `train.csv` and `test.csv`, and choose
**Runtime > Run all**. The notebook covers data exploration, baselines,
training, evaluation, and writing `submission.csv`. Then validate the
file with the organizers' script:

```bash
python utils/validator.py --file submission/submission.csv
```

The seed is fixed (42). Small run-to-run differences in GPU training are
normal, so a rerun may land a fraction of a point away from the numbers
above.

---

## Repository structure

```
.
├── ABSA_starter_notebook.ipynb   # Data, baselines, training, evaluation, demo
├── submission_verified.csv       # Validated test predictions (2,000 rows)
├── index.html                    # Project website
├── style.css                     # Website styles
├── script.js                     # Website behaviour
├── notes.md                      # Project notes: data, results, method
└── README.md
```

The organizers' dataset and validator are not included in this repository.

---

## Limitations

- **Neutral is the hard boundary.** About 85% of errors involve Neutral,
  where an aspect is mentioned without a clear opinion.
- **Single split, single seed.** Results come from one stratified
  validation split and one training run. The official test score may
  differ slightly.
- **Not tried:** the `context_with_aspect_tag` input variant, multiple
  seeds, cross-validation, ensembles, and larger models.
- **The website does not run the model.** It presents results only. The
  live demo runs from the notebook while the Colab runtime is active.

---

## Author

**Marri Ishanvi Reddy** — AI & Data Science, Year 3
Team: YOUR_TEAM_NAME (add teammates here)

GitHub: [ishanvireddy79](git@github.com:Ishanvireddy79/hackathon_team4.git) ·
Email: 2420080071@klh.edu.in