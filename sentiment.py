from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

class SentimentAnalyzer:

    def __init__(self, type: str = 'numerical', model: str = 'cardiffnlp/twitter-roberta-base-sentiment'):
        self.type = type
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.model = RobertaForSequenceClassification.from_pretrained(model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def sentiment_score_roberta(self, sentence):
        """
        Calculate the sentiment score using Roberta model.
        
        Parameters:
            sentence (str): The input sentence for sentiment analysis.
        
        Returns:
            float or str: The sentiment score. If the type is 'numerical', the score is a float representing 
            the difference between the probability of the positive class and the negative class. 
            If the type is 'binary', the score is a string representing the predicted sentiment label ('negative', 'neutral', or 'positive').
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        if self.type == 'numerical':
            sentiment_score = predictions[0][2] - predictions[0][0]
            return sentiment_score.item()

        elif self.type == 'binary':
            labels = ['negative', 'neutral', 'positive']
            sentiment_score = labels[predictions.argmax()]
            return sentiment_score

    def texts_to_sentiments(self, texts: list):
        return [self.sentiment_score_roberta(text) for text in texts]