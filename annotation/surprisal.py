import argparse
import os
import pandas as pd
import string
from annotation.SurprisalScorerLMs import SurprisalScorer


def process_tsv(file_path, scorer, stimuli_path, model="gpt2", existing_df=None, add_prompt=False):
    df = existing_df if existing_df is not None else pd.read_csv(file_path, sep='\t')
    
    if add_prompt:
        surprisal_col_name = f'surprisal_p_{model}'
    else:
        surprisal_col_name = f'surprisal_{model}'
    df[surprisal_col_name] = 0.0

    # open the file with the stimuli
    stimuli = pd.read_csv(stimuli_path, sep='\t')

    for _, group in df.groupby(['item_id', 'list', 'model', 'decoding_strategy']):

        words = group['word'].tolist()
        text = ' '.join(words)

        model = group['model'].unique().item()
        decoding_strategy = group['decoding_strategy'].unique().item()
        item_id = group['item_id'].unique().item()
        prompt = stimuli.loc[(stimuli['model']=='phi2') & (stimuli['decoding_strategy'] == 'topp') & (stimuli['item_id'] == 'item01')]['prompt'].item()

        if add_prompt:
            prompt_text = prompt + ' ' + text
            prompt_length = len(prompt.split())
        
        else:
            prompt_text = text

        surprisal_values, _ = scorer.score(prompt_text, BOS=True)

        assert len(surprisal_values) == len(prompt_text.split()), breakpoint()
        

        if add_prompt:
            surprisal_values = surprisal_values[prompt_length:]
        
        assert len(surprisal_values) == len(words), breakpoint()

        for i, word in enumerate(words):
            df.loc[group.index[i], surprisal_col_name] = surprisal_values[i]
    return df


def main(args):
    input_file = args.input_file
    output_file = args.output_file
    model = args.model

    stimuli_path = 'data/stimuli.csv'

    gpt_scorer = SurprisalScorer(model)
    existing_df = pd.read_csv(output_file, sep='\t') if os.path.exists(output_file) else None

    # compute surprisal for the text once with the prompt as context for the text
    output_df = process_tsv(
        file_path=input_file,
        scorer=gpt_scorer,
        stimuli_path=stimuli_path,
        model=model,
        existing_df=existing_df,
        add_prompt=True,
    )

    # compute surprisal 'traditionally', without the prompt as context
    output_df = process_tsv(
        file_path=input_file,
        scorer=gpt_scorer,
        stimuli_path=stimuli_path,
        model=model,
        existing_df=output_df,
        add_prompt=False,
    )

    output_df.to_csv(output_file, sep='\t', index=False)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TSV file for surprisal analysis.")
    parser.add_argument("input_file", type=str, help="Path to the input TSV file")
    parser.add_argument("output_file", type=str, help="Path to the output TSV file")
    parser.add_argument("model", type=str, help="Model to use for surprisal analysis")

    args = parser.parse_args()
    main(args)
