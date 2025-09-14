import torch
import logging
from transformers import BertTokenizer
from src.model import BertForPunctuation
import re
logging.basicConfig(level=logging.INFO)


class PunctuationRestorer:
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Loading model from {model_path} to {self.device}")

        # must match training
        self.class_to_punc = {
            0: "",  # no punctuation
            1: ",",
            2: ".",
            3: "?",
            4: "!",
            5: ":",
            6: ";"
        }

        self.model = BertForPunctuation(num_punct_classes=len(self.class_to_punc))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(self.model.classifier.weight[:3])  # just check a few rows


    def restore_punctuation(self, text, max_length=128):
        tokens = self.tokenizer(text, return_tensors="pt",
                                max_length=max_length,
                                truncation=True,
                                padding='max_length')
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            preds = outputs.argmax(dim=-1).cpu().numpy()[0]

        token_list = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        restored_text = []
        restored_text = []
        current_word = ""

        for token, pred in zip(token_list, preds):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if token.startswith("##"):
                # continuation of the previous word
                current_word += token[2:]
            else:
                # push previous word + punctuation
                if current_word:
                    restored_text.append(current_word + self.class_to_punc[last_pred])
                current_word = token
            last_pred = pred

        # append last word
        if current_word:
            restored_text.append(current_word + self.class_to_punc[last_pred])


        text_out = " ".join(restored_text).replace(" ##", "")

        # ----------- CLEAN POST-PROCESSING ----------- #
        # 1. Remove spaces before punctuation
        text_out = re.sub(r"\s+([,.!?;:])", r"\1", text_out)

        # 2. Capitalize first letter of sentence after . ? !
        sentences = re.split(r"([.?!])", text_out)
        sentences = [s.strip().capitalize() for s in sentences if s.strip() != ""]
        text_out = " ".join(sentences)

        # 3. Fix double spaces
        text_out = re.sub(r"\s+", " ", text_out).strip()
        # --------------------------------------------- #

        return text_out


if __name__ == "__main__":
    restorer = PunctuationRestorer(model_path="checkpoints/best_model.pt")
    unpunctuated_text = "i am feeling sad today how should i cope"
    result = restorer.restore_punctuation(unpunctuated_text)
    print("Restored Text:")
    print(result)
