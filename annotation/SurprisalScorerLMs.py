import torch
import numpy as np
import string
from transformers import AutoModelForCausalLM, AutoTokenizer

STOP_CHARS_SURP = []

class SurprisalScorer:
    def __init__(self, model_name="gpt2"):

        self.STRIDE = 256
        self.MAX_LENGTH = 1024

        self.name = model_name

        # load model and tokenizer
        if self.name == 'mistral-instruct':
            self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
            self.model = AutoModelForCausalLM.from_pretrained(
                'mistralai/Mistral-7B-Instruct-v0.1',
                device_map='auto',
            )
        elif self.name == 'mistral-base':
            self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
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
            accumulated_length += token_length

            if word_index < len(words):
                prob_list[word_index] *= p

            next_word_boundary = len(words[word_index]) + (word_index * 1)  # Space adjustment

            if word_index + 1 < len(words) and words[word_index + 1] in string.punctuation:
                next_word_boundary += len(words[word_index + 1]) + 1

            if accumulated_length >= next_word_boundary:
                word_index += 1
                if word_index < len(words) and words[word_index] in string.punctuation:
                    word_index += 1

            if word_index >= len(words):
                break

        assert len(prob_list) == len(words), "Mismatch in probabilities and words count"

        return prob_list
    

    def multiply_subword_metrics(self, offset, probs, text, words):
        prob = []
        j = 0
        for i in range(0, len(words)):  # i index for reference word list
            try:
                # case 1: tokenized word = white-space separated word
                # print(f'{words[i]} ~ {text[offset[j][0]:offset[j][1]]}')
                if words[i] == text[offset[j][0]: offset[j][1]].strip().lstrip():
                    prob += [probs[j]]  # add probability of word to list
                    #breakpoint()
                    # print('subword = word')
                    # print(words[i])
                    # print(text[offset[j][0]: offset[j][1]])
                    # print(text[offset[j][0]: offset[j][1]].strip().lstrip())
                    j += 1
                    
                    
                # case 2: tokenizer split subword tokens: merge subwords and add up probabilities until the same
                else:

                    #print('subword != word')
                    concat_token = text[offset[j][0]: offset[j][1]].strip().lstrip()
                    concat_prob = probs[j]

                    if j > 1:
                        if offset[j] == offset[j-1]:
                            j += 1
                            continue
 
                    #print(concat_token)
                    while concat_token != words[i]:

                        if offset[j+1] == offset[j]:
                            j += 1
                            continue
                        
                        j += 1
                        

                        #print(j)
                        concat_token += text[
                                        offset[j][0]: offset[j][1]
                                        ].strip()
                        # define characters that should not be added to word probability values
                        if (
                                text[offset[j][0]: offset[j][1]].strip().lstrip()
                                not in STOP_CHARS_SURP
                        ):
                            concat_prob *= probs[j]  # multiply probabilities
                        #print(text[offset[j][0]: offset[j][1]].strip())
                        #print(j)
                        #print(concat_token)
                    prob += [concat_prob]
                    j += 1
                    #print(j)
            except IndexError:
                #print('error')
                if len(prob) == len(words)-1:
                    prob += [concat_prob]
                break
        #breakpoint()
        assert len(prob) == len(words), f"Length of probabilities ({len(prob)}) does not match length of words ({len(words)}) for sentence {sent}"
        return prob


    def score(self, text_seq, BOS=True):
        with torch.no_grad():
            words = text_seq.split()
            all_probs = torch.tensor([], device=self.model.device)
            start_ind = 0
            offset_mapping = []

            while True:
                encodings = self.tokenizer(
                    text_seq[start_ind:],
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

                offset = 0 if start_ind == 0 else self.STRIDE - 1
                all_probs = torch.cat([all_probs, subtoken_probs[offset:-1]])
                #offset_mapping = [(i + start_ind, j + start_ind) for i, j in encodings["offset_mapping"][1:-1]]
                #offset_mapping = [(i + start_ind, j + start_ind) for i, j in encodings["offset_mapping"]]
                offset_mapping.extend(
                    [
                        (i + start_ind, j + start_ind)
                        for i, j in encodings["offset_mapping"][offset:]
                    ]
                )


                if encodings["offset_mapping"][-1][1] + start_ind >= len(text_seq):
                    break

                start_ind += encodings["offset_mapping"][-self.STRIDE][1]

            #breakpoint()
            #prob_list = self.add_subword_metrics(offset_mapping, all_probs.cpu().numpy(), text_seq, words)
            prob_list = self.multiply_subword_metrics(offset_mapping, all_probs.cpu(), text_seq, words)

            # if not len(prob_list) == len(words):
            #     breakpoint()

            assert len(prob_list) == len(words), "Mismatch in probabilities and words count"
            #surprisal_values = -np.log2(np.clip(prob_list, a_min=5e-10, a_max=None))  # Prevent log(0)
            surprisal_values = -np.log(np.clip(prob_list, a_min=5e-10, a_max=None))

            if 0.0 in surprisal_values:
                print(f"Warning: Zero surprisal values found for text_seq: '{text_seq}'")
                print(f"Probabilities: {prob_list}")
                print(f"Surprisal values: {surprisal_values}")
                print(f"Words: {words}")

            return np.asarray(surprisal_values), len(words)