import glob
import os
import pandas as pd


def main():
    path_to_phi = '../../output_generation/phi2/2024-01-08/'
    path_to_mistral = '../../output_generation/mistral/2024-01-07/'
    # path_to_llama30 = '../output_generation/llama30/2024-01-05/'
    path_to_wizardlm = '../../output_generation/wizardlm/2024-01-07/'
    # paths = [path_to_phi, path_to_mistral, path_to_llama30, path_to_wizardlm]
    paths = [path_to_phi, path_to_mistral, path_to_wizardlm]

    phi_topp = pd.read_csv(os.path.join(path_to_phi, 'phi2_decoding-topp_150_items-0-66.csv'))
    item_ids = phi_topp['item_id'].tolist()
    prompts = phi_topp['prompt']

    new_dict = {
        'item_id': item_ids,
        'prompt': prompts,
    }
    for path in paths:
        files = os.listdir(path)
        for file in files:
            if not '-0-66' in file:
                continue
            new_dict[file] = list()
            out = pd.read_csv(os.path.join(path, file))
            for item_id in new_dict['item_id']:
                if file == 'llama30_decoding-beam_search_150_items-0-66.csv' and item_id in ['item43', 'item44']:
                    new_dict[file].append('NaN')
                else:
                    try:
                        new_dict[file].append(
                            out.loc[out['item_id'] == item_id, 'gen_seq_trunc'].tolist()[0]
                        )
                    except IndexError:
                        new_dict[file].append('NaN')

    new_dict_df = pd.DataFrame(new_dict)
    new_dict_df.to_csv('all_gen_seqs_trunc_02.csv')


if __name__ == '__main__':
    main()
