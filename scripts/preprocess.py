import pandas as pd
train_file = "dataset/train.txt"
val_file = "dataset/val.txt"
test_file = "dataset/test.txt"

def load_dataset(file_path, separator=";"):
    texts = []
    labels = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(separator)
            if len(parts) == 2:  
                texts.append(parts[0])  
                labels.append(parts[1]) 
    
    return pd.DataFrame({"text": texts, "label": labels})

train_df = load_dataset(train_file)
val_df = load_dataset(val_file)
test_df = load_dataset(test_file)
train_df.to_csv("dataset/train.csv", index=False)
val_df.to_csv("dataset/val.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

print("✅ Dataset Preprocessed and Saved!")
