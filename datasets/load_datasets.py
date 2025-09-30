# Load QEvasion dataset, keep useful columns, and save as CSV.
# gpt3.5_summary and gpt3.5_prediction cause issues, so left commented out.


from datasets import load_dataset

dataset = load_dataset("ailsntua/QEvasion")

train_dataset = dataset['train']
test_dataset = dataset['test']

keep_cols = [
'title', 'date', 'president', 'url', 'question_order',
       'interview_question', 'interview_answer', 
    #  'gpt3.5_summary', 
    #  'gpt3.5_prediction', 
       'question', 'annotator_id', 'annotator1',
       'annotator2', 'annotator3', 'inaudible', 'multiple_questions',
       'affirmative_questions', 'index', 'clarity_label', 'evasion_label'
]
train_df = train_dataset.to_pandas()
print(train_df.columns)
train_df = train_df[keep_cols]
test_df = test_dataset.to_pandas()

train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)