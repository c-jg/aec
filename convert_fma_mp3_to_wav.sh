#!/bin/bash

read -p 'Enter path to fma data: '  INPUT_PATH

read -p 'Enter path to save wav files to: ' OUTPUT_PATH

ALL_MP3_FILES=$(find $INPUT_PATH -name '*.mp3' -type f)

for i in $ALL_MP3_FILES; do
	FILE_NAME=$(basename $i .mp3)
	OUTPUT_WAV="$OUTPUT_PATH/$FILE_NAME.wav"
	ffmpeg -i $i -ar 16000 -ac 1 $OUTPUT_WAV
done;

echo "Done converting mp3 files to 16kHz mono wav files."
