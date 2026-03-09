#!/bin/bash


# Run the whisper_main.py script and capture only stderr
output=$(python3 Whisper.py 2>&1 >/dev/null)
if echo "$output" | grep -q "KeyError: 'speaker'"; then
    # Exit with status 1 if KeyError is found
    exit 1
else
    # Exit with status 0 if no error is found
    exit 0
fi
