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
        enhanced_data = []

        for _, row in self.data.iterrows():
            item_aoi = self.aoi_data[
                (self.aoi_data['item_id'] == row['item_id']) &
                (self.aoi_data['model'] == row['model']) &
                (self.aoi_data['decoding_strategy'] == row['decoding_strategy'])
            ]

            text = " ".join(item_aoi['word'].tolist())
            doc = nlp(text)
            tokens = [(token.text, token) for token in doc]

            word_data = []
            original_words = text.split()
            original_index = 0
            i = 0

            while i < len(tokens) and original_index < len(original_words):
                current_word = tokens[i][0]
                annotations = [tokens[i][1]]
                current_y_top = item_aoi.iloc[i]['y_top'] if i < len(item_aoi) else None

                # Concatenate tokens until the joined string matches the current original word
                while (current_word != original_words[original_index] and i < len(tokens) - 1):
                    i += 1
                    current_word += tokens[i][0]
                    annotations.append(tokens[i][1])

                # Determine if it's the last in line based on y_top change or document end
                last_in_line = 0
                if i < len(tokens) - 1:
                    next_y_top = item_aoi.iloc[i + 1]['y_top'] if i + 1 < len(item_aoi) else None
                    if current_y_top != next_y_top:
                        last_in_line = 1
                else:
                    last_in_line = 1  # Mark as last in line if it's the last token

                # Verify match with the original word
                if current_word == original_words[original_index]:
                    # Aggregate annotations
                    aggregated = {
                        'item_id': row['item_id'],
                        'model': row['model'],
                        'decoding_strategy': row['decoding_strategy'],
                        'list': row['list'],
                        'word_index': original_index,
                        'word': current_word,
                        'POS': annotations[0].pos_ if len(annotations) == 1 else [ann.pos_ for ann in annotations],
                        'dependency_tag': annotations[0].dep_ if len(annotations) == 1 else [ann.dep_ for ann in annotations],
                        'n_dep_left': sum(len(list(ann.lefts)) for ann in annotations),
                        'n_dep_right': sum(len(list(ann.rights)) for ann in annotations),
                        'distance_to_head': sum(abs(ann.i - ann.head.i) for ann in annotations),
                        'word_length_with_punct': sum(len(ann.text) for ann in annotations),
                        'word_length_without_punct': sum(len(re.sub(r'\W', '', ann.text)) for ann in annotations),
                        'last_in_line': last_in_line
                    }
                    word_data.append(aggregated)
                    original_index += 1

                i += 1

            # Verify the number of processed words matches the original text words
            if len(word_data) != len(original_words):
                print(f"Warning: Mismatch in word counts for item_id {row['item_id']}")

            enhanced_data.extend(word_data)

        return pd.DataFrame(enhanced_data)

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
