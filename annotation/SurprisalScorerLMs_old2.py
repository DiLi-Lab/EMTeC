import torch
import numpy as np
import string
from transformers import AutoModelForCausalLM, AutoTokenizer

class SurprisalScorer:
    def __init__(self, model_name="gpt2"):

        self.STRIDE = 256
        self.MAX_LENGTH = 1024

        self.name = model_name

        # load model and tokenizer
        if self.name == 'mistral':
            self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
            self.model = AutoModelForCausalLM.from_pretrained(
                'mistralai/Mistral-7B-v0.1',
                device_map='auto',
            )
        elif self.name == 'phi2':
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
            self.model = AutoModelForCausalLM.from_pretrained(
                'microsoft/phi-2',
                device_map='auto',
                trust_remote_code=True,
                torch_dtype='auto',
            )
        elif self.name == 'llama2-7b':
            self.tokenizer = AutoTokenizer.from_pretrained('/srv/scratch3/llm/Llama-2-7b-hf')
            self.model = AutoModelForCausalLM.from_pretrained(
                '/srv/scratch3/llm/Llama-2-7b-hf',
                device_map='auto',
            )
        elif self.name == 'llama2-13b':
            self.tokenizer = AutoTokenizer.from_pretrained('/srv/scratch3/llm/Llama-2-13b-hf')
            self.model = AutoModelForCausalLM.from_pretrained(
                '/srv/scratch3/llm/Llama-2-13b-hf',
                device_map='auto',
            )
        elif self.name == 'gpt2':
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.model = AutoModelForCausalLM.from_pretrained(
                'gpt2',
                device_map='auto',
            )
        elif self.name == 'gpt2-large':
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
            self.model = AutoModelForCausalLM.from_pretrained(
                'gpt2-large',
                device_map='auto',
            )
        elif self.name == 'opt-350m':
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
            self.model = AutoModelForCausalLM.from_pretrained(
                'facebook/opt-350m',
                device_map='auto',
            )
        elif self.name == 'opt-1.3b':
            self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
            self.model = AutoModelForCausalLM.from_pretrained(
                'facebook/opt-1.3b',
                device_map='auto',
            )
        elif self.name == 'pythia-6.9b':
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-6.9b')
            self.model = AutoModelForCausalLM.from_pretrained(
                'EleutherAI/pythia-6.9b',
                device_map='auto',
            )
        elif self.name == 'pythia-12b':
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-12b')
            self.model = AutoModelForCausalLM.from_pretrained(
                'EleutherAI/pythia-12b',
                device_map='auto',
            )
        else:
            raise NotImplementedError(f'Surprisal extraction for model {self.name} is not implemented.')

        self.model.eval()


    def add_subword_metrics(self, offset, probs, sent, words):
        prob_list = [1.0] * len(words)  # Initialize probabilities for each word as 1 (100%)
        word_index = 0  # Start with the first word
        accumulated_length = 0  # Track accumulated length to compare with word lengths
        token_lengths = [end - start for start, end in offset]  # Length of each token

        for token_length, p in zip(token_lengths, probs):
            # Increase accumulated length by the token's length
            accumulated_length += token_length

            # Multiply the probability to the current word
            if word_index < len(words):
                prob_list[word_index] *= p

            # Prepare to move to the next word
            # Check if the accumulated length has reached or exceeded the length of the current word
            next_word_boundary = len(words[word_index]) + (word_index * 1)  # Adjusted to handle spaces more accurately

            # Check if this token is a punctuation that should be concatenated with the current word
            if word_index + 1 < len(words) and words[word_index + 1] in string.punctuation:
                next_word_boundary += len(words[word_index + 1]) + 1  # Include the punctuation length and space

            if accumulated_length >= next_word_boundary:
                word_index += 1
                # Skip the increment of word_index if the next token is punctuation to be concatenated
                if word_index < len(words) and words[word_index] in string.punctuation:
                    word_index += 1

            if word_index >= len(words):  # Ensure the index does not exceed the list
                break

        # Debug output for each word and its corresponding probability
        for idx, word in enumerate(words):
            print(f"{idx} {word} {prob_list[idx]}")
        print('-----------------------------------')

        print(f"words: {words}, prob: {prob_list}")
        assert len(prob_list) == len(words), f"Length of probabilities ({len(prob_list)}) does not match length of words ({len(words)}) for sentence '{sent}'"
        return prob_list

    def score(self, sentence, BOS=True):
        with torch.no_grad():
            words = sentence.split()
            all_probs = torch.tensor([], device=self.model.device)
            start_ind = 0

            while True:
                encodings = self.tokenizer(
                    sentence[start_ind:],
                    max_length=self.MAX_LENGTH - 2,  # Account for potential BOS/EOS
                    truncation=True,
                    return_offsets_mapping=True
                )
                tensor_input = torch.tensor(
                    [([self.tokenizer.bos_token_id] if BOS else []) + encodings["input_ids"] + [self.tokenizer.eos_token_id]],
                    device=self.model.device
                )

                output = self.model(tensor_input, labels=tensor_input)
                logits = output.logits[..., :-1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                labels = tensor_input[..., 1:].contiguous()

                subtoken_probs = probs[0, torch.arange(labels.size(-1)), labels[0]]
                all_probs = torch.cat([all_probs, subtoken_probs])
                offset_mapping = [(i + start_ind, j + start_ind) for i, j in encodings["offset_mapping"][1:-1]]

                if encodings["offset_mapping"][-1][1] + start_ind >= len(sentence):
                    break

                start_ind += encodings["offset_mapping"][-self.STRIDE][1]

            prob_list = self.add_subword_metrics(offset_mapping, all_probs.cpu().numpy(), sentence, words)
            surprisal_values = -np.log2(np.clip(prob_list, a_min=5e-10, a_max=None))  # Prevent log(0)
            if 0.0 in surprisal_values:
                print(f"Warning: Zero surprisal values found for sentence: '{sentence}'")
                print(f"Probabilities: {prob_list}")
                print(f"Surprisal values: {surprisal_values}")
                print(f"Words: {words}")

            return np.asarray(surprisal_values), len(words)
