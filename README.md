# Word-Mover-Distance
You can use WMD to evaluate one caption or diverse captions for one image. Given an image, the corresponding human annotations/annotation and the generate captions/caption, we first tokenize human annotations/annotation and obtain a dictionary ![](http://latex.codecogs.com/gif.latex?\\mathcal{D}) and then tokenize the generated captions/caption and obtain another dictionary ![](http://latex.codecogs.com/gif.latex?\\hat{\mathcal{D}}), finally, we compute the Word Mover Distance between the two dictionaries.
## How use the metric?
1. Download the trained word2vec models and put in the *trained_models/word2vec* folder.
    1. WMD could be highly related to the corpus. We provide a word2vec model trained on MSCOCO captions ([Download here](https://drive.google.com/drive/folders/1vW0xr14TKiQBNXIe_N3HLpCY0Xx8m7W1)). Alternatively, you can use other trained word2vec models, such as GoogleNews model. 
2. Download the tokenized MSCOCO dataset and put it in the *data/files* folder.
    1. You can [download here](https://drive.google.com/drive/folders/1qm85vYouLJMYTjESscglcKVZQDASMuck) (download all 3 files).
3. Use your method to generate captions/caption and save as a json file, the format of which must be the same as *results/results_bs3.json* (each image has one caption) or *results/merge_results10.json* (each image has 10 captions).
4. Run the command ```python WMD_accuracy.py --results_file ../results/merge_results10.json --score_file ../results/merge_results10_score.json --num_captions 10 --exp 1```
## References
* Matt J. Kusner et. al., From Word Embeddings To Document Distances. ICML, 2015.
* Mert Kilickaya et. al., Re-evaluating Automatic Metrics for Image Captioning. EACL, 2017.
