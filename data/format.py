import csv
import nltk
from nltk.tokenize import sent_tokenize

# Ensure necessary resources are downloaded
nltk.download('punkt')

def split_text_strict(text, max_words=100):
    sentences = sent_tokenize(text)
    rows = []
    current_row = []
    word_count = 0

    for sentence in sentences:
        words_in_sentence = sentence.split()
        for word in words_in_sentence:
            current_row.append(word)
            word_count += 1
            if word_count >= max_words:
                rows.append(' '.join(current_row))
                current_row = []
                word_count = 0

    if current_row:
        rows.append(' '.join(current_row))

    return rows

# Read input with utf-8
with open('input_text.txt', 'r', encoding='utf-8', errors='ignore') as file:
    text = file.read()

rows = split_text_strict(text, max_words=100)

# Write CSV with utf-8
with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in rows:
        writer.writerow([row])
