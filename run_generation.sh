#!/bin/sh

# run all models with all decoding strategies on all prompts

### Phi2

CUDA_VISIBLE_DEVICES=0,1 python -m generation.generate --dec sampling --model phi2 --subset-idx 0 4
CUDA_VISIBLE_DEVICES=0,1 python -m generation.generate --dec greedy_search --model phi2
CUDA_VISIBLE_DEVICES=0,1 python -m generation.generate --dec topk --model phi2
CUDA_VISIBLE_DEVICES=0,1 python -m generation.generate --dec topp --model phi2


### Mistral

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m generation.generate --dec sampling --model mistral
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m generation.generate --dec topk --model mistral
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m generation.generate --dec topp --model mistral
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m generation.generate --dec greedy_search --model mistral
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m generation.generate --dec beam_search --model mistral


### WizardLM

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec sampling
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec topk
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec topp
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec greedy_search
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m generation.generate --model wizardlm --dec beam_search

# Beam Search might result in CUDA OOM Errors; then generate for groups of prompts individually
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate--model wizardlm --dec beam_search --subset-idx 0 36
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 36 37
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 37 38
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 38 39
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 39 40
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 40 42
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 42 43
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 43 44
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 44 45
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 45 50
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 50 65
#CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m generation.generate --model wizardlm --dec beam_search --subset-idx 66 105

