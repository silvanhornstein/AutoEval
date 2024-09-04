# Predicting Satisfaction with Chat-Counseling at a 24/7 Chat Hotline for the Youth: A Natural Language Processing Study  
Code for the Paper, submitted to JMIR AI (https://ai.jmir.org/)
### Silvan Hornstein, Ulrike Lueken, Richard Wundrack, Kevin Hilbert
### Abstract
Background: Chat-based counseling services are popular for the low-threshold provision of mental health support to the youth. Also, they are particularly suitable for the utilization of Natural Language Processing (NLP) for an improved provision of care. 

Objective:  Consequently, this paper evaluates the feasibility of such a use case, namely the NLP-based automated evaluation of satisfaction with the chat interaction. This preregistered (OSF: SR4Q9) approach could be utilized for evaluation and quality control procedures, as being particularly relevant for those services.

Methods: The consultations of 2,609 young chatters (around 140,000 messages) and corresponding feedback were used to train and evaluate classifiers to predict whether a chat was perceived as helpful or not. On the one hand, we trained a word-vectorizer in combination with a XGBoost classifier, applying cross-validation and extensive hyperparameter tuning. On the other hand, we trained several transformer-based models, comparing model-types, preprocessing and over- and under sampling techniques. For both model types, we selected the best performing approach on the training set for a final performance evaluation on the 522 users in the final test set. 

Results: The fine-tuned XGBoost classifier achieved an AUROC score of 0.69 (P <.0 01) as well as an MCC of 0.25 on the previously unseen test set. The selected Longformer-based model did not outperform this baseline, scoring 0.68 (P = .69). A SHAP explainability approach suggested that help seekers rating a consultation as helpful commonly expressed their satisfaction already within the conversation. In contrast, the rejection of offered exercises predicted perceived unhelpfulness. 

Conclusions: Chat conversations include relevant information regarding the perceived quality of an interaction that can be utilized by NLP-based prediction approaches. However, to determine if the moderate predictive performance translates into meaningful service improvements requires randomized trials. Further, our results highlight the relevance of contrasting pretrained models with simpler baselines to avoid the implementation of unnecessarily complex models. 

Keywords: Natural Language Processing (NLP), Artificial Intelligence (AI), Digital Mental Health (DMH), Adolescence 

### Links:

OSF Registration: https://osf.io/sr4q9 <br/>
Preprint: (https://preprints.jmir.org/preprint/63701)
