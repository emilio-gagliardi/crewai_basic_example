#!/bin/bash
model_name="llama3"
custom_model_name="crewai-llama3:8b"

# Pull the base model
ollama pull $model_name

# Create the custom model using the ModelFile
ollama create $custom_model_name -f ./Llama3ModelFile
