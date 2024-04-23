#!/bin/bash
set -x

# create the files with AOIs for each combination of model/decoding strategy/item id
echo "Creating unique aoi csv files ..."
python -m annotation.create_aoi_csvs

# Assign arguments to variables
INPUT_FILE="data/stimuli.csv"
AOI_DIRECTORY="data/unique_aois"
OUTPUT_FILE="annotation/word_level_annotations.csv"
shift 3  # Shift the first three arguments to get model names
MODELS=("$@")

# Path to the Python scripts
#SURPRISAL_SCRIPT="/Users/isabellecretton/Desktop/WARFARE/ET_DECODING/annotation/surprisal.py"
#ANNOTATIONS_SCRIPT="/Users/isabellecretton/Desktop/WARFARE/ET_DECODING/annotation/annotations.py"
SURPRISAL_SCRIPT="annotation.surprisal"
ANNOTATIONS_SCRIPT="annotation.annotations"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Input file $INPUT_FILE not found!"
    exit 1
fi

# Check if AOI directory exists
if [ ! -d "$AOI_DIRECTORY" ]; then
    echo "AOI directory $AOI_DIRECTORY not found!"
    exit 1
fi

# Temporary output file for the annotations script
TEMP_OUTPUT="annotation/temp_annotated_data.csv"

# Run annotations script
echo "Running annotations enrichment..."
python -m "$ANNOTATIONS_SCRIPT" "$INPUT_FILE" "$AOI_DIRECTORY" "$TEMP_OUTPUT"

# Check if the annotations script ran successfully
if [ ! -f "$TEMP_OUTPUT" ]; then
    echo "Annotations enrichment failed or output file $TEMP_OUTPUT not created."
    exit 1
fi

# Run surprisal script for each model
echo "Running surprisal analysis for each model..."
for MODEL in "${MODELS[@]}"; do
    echo "Processing with model: $MODEL"
    CUDA_VISIBLE_DEVICES=5,6,7 python -m "$SURPRISAL_SCRIPT" "$TEMP_OUTPUT" "$OUTPUT_FILE" "$MODEL"
done

# Check if the surprisal script ran successfully and output file was created
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Surprisal analysis failed or output file $OUTPUT_FILE not created."
    rm "$TEMP_OUTPUT"  # Clean up temporary file in case of failure
    exit 1
fi

# Cleanup temporary files if everything went well
rm "$TEMP_OUTPUT"

echo "Processing completed. Output file generated at $OUTPUT_FILE"
