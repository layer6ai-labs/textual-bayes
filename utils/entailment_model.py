import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EntailmentDeberta:
    def __init__(self, cuda=False):
        if cuda:
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        cache_dir = "./models/microsoft/deberta-v2-xlarge-mnli"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(cache_dir).to(
                self.device
            )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/deberta-v2-xlarge-mnli", cache_dir=cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v2-xlarge-mnli", cache_dir=cache_dir
            ).to(self.device)

    def check_implication(self, text1, text2, output_probs=False, *args, **kwargs):
        """
        The model checks if `text1` entails `text2`.
        If the model predicts 2 - `entailment`, then `text1` entails `text2`.
        If the model predicts 1 - `neutral`, then `text1` does not entail `text2`.
        If the model predicts 0 - `contradiction`, then `text1` contradicts `text2`.
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.squeeze(F.softmax(logits, dim=1))
        if output_probs:
            return probs.cpu().data.numpy()
        largest_index = torch.argmax(probs)
        prediction = largest_index.cpu().item()

        return prediction
