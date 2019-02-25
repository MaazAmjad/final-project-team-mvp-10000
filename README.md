# Harvey Mudd College at SemEval-2019 Task 4: The Clint Buchanan Hyperpartisan News Detector

This repo contains the training, validation, and experimental code used in our submission to the [SemEval 2019 Task 4: Hyperpartisan News Detection](https://pan.webis.de/semeval19/semeval19-web/) where our model came in **10th place** internationally with an accuracy of **77.1%** according to the leaderboard available [here](https://pan.webis.de/semeval19/semeval19-web/leaderboard.html).

Team Clint Buchanan:
- Mehdi Drissi
- Pedro Sandoval Segura
- Vivaswat Ojha

Code for text preprocessing was provided to us by Professor Julie Medero as part of a course in Natural Language Processing. She was instrumental in helping us submit our model to the competition and write our workshop paper. This code can be found in `extract_articles.py`, `extract_features.py`, `predict.py`, `preprocess.py`.

We trained Bidirectional Encoder Representations from Transformers (BERT) models (Devlin et al., 2018) based on the implementation from [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT). Please refer to that repository for required dependencies. 


To train our BERT model, you can use the `train.sh` bash script. You will need to download the articles [here](https://zenodo.org/record/1489920#.XHN7Ds9Kiu4).






