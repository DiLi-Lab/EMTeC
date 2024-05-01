import argparse
import os
import pandas as pd
import string
from annotation.SurprisalScorerLMs import SurprisalScorer

def process_tsv(file_path, scorer, model="gpt2", existing_df=None):
    df = existing_df if existing_df is not None else pd.read_csv(file_path, sep='\t')
    surprisal_col_name = f'surprisal_{model}'
    df[surprisal_col_name] = 0.0

    for _, group in df.groupby(['item_id', 'list', 'model', 'decoding_strategy']):
        # Create text where punctuation is attached to the preceding word
        words = group['word'].tolist()
        processed_text = []
        for word in words:
            # safeguard to
            if word in string.punctuation:
                if processed_text:
                    processed_text[-1] += word
            else:
                processed_text.append(word)
        text = ' '.join(processed_text)

        surprisal_values, _ = scorer.score(text, BOS=True)

        # Assign surprisal values to DataFrame using original indexing
        j = 0  # index for surprisal_values
        for i, word in enumerate(words):
            if word not in string.punctuation:
                df.loc[group.index[i], surprisal_col_name] = surprisal_values[j]
                j += 1
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
