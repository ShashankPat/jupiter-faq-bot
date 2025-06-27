## eval_set.py for the evaluation 
import pandas as pd

df = pd.read_csv("jupiter_all_qa_clean.csv")

# Sample 20 random rows 
eval_df = df.sample(n=20, random_state=123)[["question","answer"]]
eval_df = eval_df.rename(columns={"question":"query","answer":"gold_answer"})

eval_df.to_csv("eval_set.csv", index=False)
print("Randomly sampled 20 rows → eval_set.csv")
