# Applying Natural Language Processing for the Automated Evaluation of Chat-Counseling Sessions for the Youth
Code for the Paper, submitted to JMIR AI (https://ai.jmir.org/)
### Silvan Hornstein, Ulrike Lueken, Richard Wundrack, Kevin Hilbert
### Abstract
Background: 
Chat-based counseling services are popular for the low-threshold provision of mental health support to the youth. Also, they are particularly suitable for the utilization of Natural Language Processing (NLP) for an improved provision of care.

Objective: 
Consequently, this paper evaluates the feasibility of such an use case, namely the NLP-based automated evaluation of satisfaction with the chat interaction. This preregistered (OSF: SR4Q9) approach could be utilized for evaluation and quality control procedures, as being particularly relevant for those services. 

Methods:
The consultations of 2,609 young chatters (around 140,000 messages) and corresponding feedback were used to train and evaluate classifiers to predict whether a chat was perceived as helpful or not. On the one hand, we trained a word-vectorizer in combination with a XGBoost classifier, applying cross-validation and extensive hyperparameter tuning. On the other hand, we trained several transformer-based models, comparing model-types, preprocessing and over- and undersampling techniques. For both model types, we selected the best performing approach on the training set for a final performance evaluation on the 522 users in the final test set. 

Results: 
The fine tuned XGBoost classifier achieved an AUROC score of 0.67 (p < . 01) on the previously unseen test set. The selected longformer-based model did not outperform this baseline, scoring 0.67 as well (p > .9). A SHAP explainability approach suggested that help seekers rating a consultation as helpful commonly expressed their satisfaction already within the conversation. In contrast, the rejection of offered exercises predicted perceived unhelpfulness. 

Conclusions:
Chat conversations include relevant information regarding the perceived quality of an interaction that can be utilized by NLP-based prediction approaches. However, to determine if the moderate predictive performance translates into meaningful service improvements requires randomized trials. Further, our results highlight the relevance of contrasting pretrained models with simpler baselines to avoid the implementation of unnecessarily complex models. 

 Keywords: Natural Language Processing (NLP), Artificial Intelligence (AI), Digital Mental Health (DMH), Adolescence 

### This Repository contains
 1 . Algorithm Training Code, for the TFIDF as well as the Transformer Approach <br/>
 2.  Code for the Algorithm comparison and evaluation, applying the statistical tests: evaluation.ipynb <br/>
 3. Code for the explainability approach <br/>
 4. utils.py with the used functions for the evaluation <br/>

### Links:

OSF Registration: https://osf.io/sr4q9 <br/>
Preprint: to be added
