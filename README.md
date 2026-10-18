# Evaluating AI Models on Maritime Data Analysis

Can a small open-source AI model be trusted to answer business questions about a shipping fleet by writing its own Pandas code? And does a better prompt make it more reliable?

This project builds a benchmark to find out.

## What it does

1. Loads the [Ship Performance Clustering Dataset](https://www.kaggle.com/datasets/jeleeladekunlefijabi/ship-performance-clustering-dataset) from Kaggle: 2,736 voyage records for four ship types in the Gulf of Guinea.
2. Defines **20 business questions** at three difficulty levels. Each has a reference Pandas solution that produces the ground truth.
3. Asks two open-source models (Qwen2.5-Coder 0.5B and 1.5B) to answer every question by writing Pandas code, under a **baseline prompt** and an **improved prompt** (2 models × 2 prompts × 20 questions = 80 graded responses).
4. Runs each response in a restricted sandbox and **grades it automatically** as correct, wrong answer, runtime error or no answer.
5. Ranks the models head to head and exports every failure for **human review** with an error category and written rationale.
6. Exports the verified solutions as a **fine-tuning dataset** in chat format (16 training and 4 validation records).

## Results

Full results, including accuracy by model and prompt design, are in the notebook. A summary table will be added here after the final run.

## Skills shown

Python, Pandas, benchmark design, AI model evaluation, prompt engineering, error analysis, supervised fine-tuning data preparation, Hugging Face Transformers.

## How to run

1. Open `Maritime_AI_Model_Evaluation.ipynb` in Google Colab.
2. Set the runtime to **T4 GPU**.
3. Run all cells. The dataset downloads from Kaggle automatically.

Setting `DRY_RUN = True` tests the pipeline with a simulated model. Dry-run numbers are not real results.

## Limitations

- 20 questions is a small benchmark, so one question is worth 5 points.
- All questions use one dataset, and that dataset is synthetic.
- Each question is run once with greedy decoding.

## Author

Anointed Akai, Data Analyst and Marine Engineer · [LinkedIn](https://www.linkedin.com/in/akaianointed)
