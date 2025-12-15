## 1. File Description
- dataset_embedding
    - includes the embedding and dataset for electrolytic kemet capacitor
- optimizer
  - methods for finding best capacitor,using embedding and query
- query
  - finding best qualified capacitor based on  latent space

## 2. pracitce steps
- run the optimizer, find best cap for design
- run dataset_embedding/maintrain_RCMWAE to train the capacitor model
- run dataset_embedding/pretrain_MWAE to refine the pretrain model
- run query/query_example_MT_RCMixedWAE.py to evaluate the model performance

## 3. Acknowledgements / Disclaimer
Parts of the code in this repository were generated or optimized with the assistance of AI tools (e.g., ChatGPT, Gemini, Copilot). 
All AI-generated outputs have been verified and modified by the authors.