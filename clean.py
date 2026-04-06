import pandas as pd


IPIP_ITEM_COLUMNS = (
    [f"EXT{i}" for i in range(1, 11)] +
    [f"EST{i}" for i in range(1, 11)] +
    [f"AGR{i}" for i in range(1, 11)] +
    [f"CSN{i}" for i in range(1, 11)] +
    [f"OPN{i}" for i in range(1, 11)]
)

def prepare_human_data():
    """Delete columns from EXT1_E to exclude_any from the human dataset"""
    df = pd.read_csv("/Users/arsh/Documents/LLM as Human Participant/Study/data/data-cleaned.csv")
    
    # Find the starting and ending column indices
    start_col = "EXT1_E"
    end_col = "exclude_any"
    
    if start_col in df.columns and end_col in df.columns:
        start_idx = df.columns.get_loc(start_col)
        end_idx = df.columns.get_loc(end_col)
        
        # Drop columns from EXT1_E to exclude_any (inclusive)
        cols_to_drop = df.columns[start_idx:end_idx+1]
        df = df.drop(columns=cols_to_drop)
        
        # Save the result
        df.to_csv("/Users/arsh/Documents/LLM as Human Participant/Study/data/human data.csv", index=False)
        
        print(f"Deleted {len(cols_to_drop)} columns from human dataset")
    else:
        print("Columns not found in human dataset")

def prepare_llm_data():
    """Delete agent_id and model_name columns from the LLM dataset"""
    df = pd.read_csv("/Users/arsh/Documents/LLM as Human Participant/Study/data/llm_answers.csv")
    
    cols_to_drop = [col for col in ['agent_id', 'model_name'] if col in df.columns]
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        df.to_csv("/Users/arsh/Documents/LLM as Human Participant/Study/data/llm_data.csv", index=False)
        print(f"Deleted columns from LLM dataset: {cols_to_drop}")
    else:
        print("Columns not found in LLM dataset")


if __name__ == "__main__":
    prepare_human_data(
        "data/raw_human.csv",
        "data/human_clean.csv"
    )

    prepare_llm_data(
        "data/raw_llm.csv",
        "data/llm_clean.csv"
    )