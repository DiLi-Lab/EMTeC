import argparse
import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from annotation.SurprisalScorerLMs import SurprisalScorer

# Initialize NLTK sentence tokenizer
nltk.download('punkt', quiet=True)

def process_tsv(file_path, scorer, model="gpt2", existing_df=None):
    df = existing_df if existing_df is not None else pd.read_csv(file_path, sep='\t')
    surprisal_col_name = f'surprisal_{model}'
    df[surprisal_col_name] = 0.0

    for _, group in df.groupby(['item_id', 'list']):
        text = ' '.join(group['word'].astype(str))
        surprisal_values, _ = scorer.score(text, BOS=True)

        # Assign surprisal values to DataFrame
        for i, surprisal in enumerate(surprisal_values):
            if i < len(group):
                df.at[group.index[i], surprisal_col_name] = surprisal

    return df

def main(args):
    input_file = args.input_file
    output_file = args.output_file
    model = args.model

    gpt_scorer = SurprisalScorer(model)
    existing_df = pd.read_csv(output_file, sep='\t') if os.path.exists(output_file) else None

    output_df = process_tsv(input_file, gpt_scorer, model, existing_df)
    output_df.to_csv(output_file, sep='\t', index=False)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TSV file for surprisal analysis.")
    parser.add_argument("input_file", type=str, help="Path to the input TSV file")
    parser.add_argument("output_file", type=str, help="Path to the output TSV file")
    parser.add_argument("model", type=str, help="Model to use for surprisal analysis")

    args = parser.parse_args()
    main(args)

