import argparse
import pandas as pd
import numpy as np
import spacy
import re
import os
from wordfreq import word_frequency, zipf_frequency

# Load the SpaCy NLP model for English
nlp = spacy.load("en_core_web_sm")

class Annotations:
    def __init__(self, data, aoi_data):
        self.data = data
        self.aoi_data = aoi_data

    def enhance_annotations(self):
        annotated_data = []

        for _, row in self.data.iterrows():
            item_aoi = self.aoi_data[
                (self.aoi_data['item_id'] == row['item_id']) &
                (self.aoi_data['model'] == row['model']) &
                (self.aoi_data['decoding_strategy'] == row['decoding_strategy'])
            ]
            text = " ".join(item_aoi['word'].tolist())
            doc = nlp(text)

            for word_index, token in enumerate(doc):
                is_last_in_line = 0  # Default is not the last in line

                current_y_top = item_aoi.iloc[word_index]['y_top'] if word_index < len(item_aoi) else None

                if word_index < len(doc) - 1:
                    next_token = doc[word_index + 1]
                    next_y_top = item_aoi.iloc[word_index + 1]['y_top'] if word_index + 1 < len(item_aoi) else None

                    if current_y_top != next_y_top:
                        is_last_in_line = 1
                else:
                    # If it's the last word in the text, mark it as last in line
                    is_last_in_line = 1

                token_data = {
                    'item_id': row['item_id'],
                    'model': row['model'],
                    'decoding_strategy': row['decoding_strategy'],
                    'list': row['list'],
                    'word_index': word_index,
                    'word': token.text,
                    'POS': token.pos_,
                    'dependency_tag': token.dep_,
                    'n_dep_left': len(list(token.lefts)),
                    'n_dep_right': len(list(token.rights)),
                    'distance_to_head': abs(token.i - token.head.i) if token.head else 0,
                    'word_length_with_punct': len(token.text),
                    'word_length_without_punct': len(re.sub(r'\W', '', token.text)),
                    'last_in_line': is_last_in_line
                }

                # Handle punctuation: append to previous and inherit last_in_line if necessary
                if token.is_punct and annotated_data:
                    annotated_data[-1]['word'] += token.text
                    annotated_data[-1]['word_length_with_punct'] += len(token.text)
                    annotated_data[-1]['last_in_line'] = is_last_in_line
                else:
                    annotated_data.append(token_data)

        return pd.DataFrame(annotated_data)


    def save_to_tsv(self, annotated_df, output_file):
        annotated_df.to_csv(output_file, sep='\t', index=False)
        print(f"Saved enhanced annotations to: {output_file}")

def calculate_wordfreq(annotated_df):
    """
    Calculate word frequency and Zipf frequency for each word in the annotated data.
    The word frequency is derived from the `wordfreq` library. wordfreq uses parameter "best" by default; it uses "large" for the languages where "large" is available.
    The 'large' lists cover words that appear at least once per 100 million words and are used by default if available.
    The Zipf frequency scale provides a logarithmic measure of word occurrence per billion words, facilitating comparison across common and rare terms.

    Parameters:
    annotated_df (pd.DataFrame): Dataframe containing the text data with a 'word' column.

    Returns:
    pd.DataFrame: The input dataframe with added columns for word frequency and Zipf frequency.
    """
    # Calculate word and Zipf frequencies
    annotated_df['word_freq'] = annotated_df['word'].apply(lambda w: word_frequency(w, 'en'))
    annotated_df['zipf_freq'] = annotated_df['word'].apply(lambda w: zipf_frequency(w, 'en'))

    # Calculate negative log frequency, handling cases where frequency is zero
    annotated_df['neg_log_word_freq'] = annotated_df['word_freq'].apply(lambda x: -np.log(x) if x > 0 else 0)

    return annotated_df

def main(input_file, aoi_directory, output_file):
    data = pd.read_csv(input_file, sep='\t', quotechar='"')[['model', 'decoding_strategy', 'item_id', 'list', 'gen_seq_trunc']]

    aoi_data_list = []
    for _, row in data.iterrows():
        aoi_file = os.path.join(aoi_directory, f"{row['model']}_{row['decoding_strategy']}_{row['item_id']}_coordinates.csv")
        try:
            item_aoi_data = pd.read_csv(aoi_file, sep='\t')
            item_aoi_data.columns = item_aoi_data.columns.str.strip()
            item_aoi_data['item_id'] = row['item_id']
            item_aoi_data['model'] = row['model']
            item_aoi_data['decoding_strategy'] = row['decoding_strategy']
            aoi_data_list.append(item_aoi_data)
        except pd.errors.ParserError as e:
            print(f"Error reading {aoi_file}: {e}")
            continue

    if not aoi_data_list:
        print("No AOI data loaded. Please check the files and directory.")
        return

    aoi_data = pd.concat(aoi_data_list, ignore_index=True)
    annotations = Annotations(data, aoi_data)
    annotated_df = annotations.enhance_annotations()

    enriched_annotated_df = calculate_wordfreq(annotated_df)
    annotations.save_to_tsv(enriched_annotated_df, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and annotate eye-tracking text data.")
    parser.add_argument("input_file", help="Input TSV file path")
    parser.add_argument("aoi_directory", help="Directory containing AOI files")
    parser.add_argument("output_file", help="Output TSV file path")
    args = parser.parse_args()

    main(args.input_file, args.aoi_directory, args.output_file)
