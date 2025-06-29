from nltk.tokenize import sent_tokenize

text = "The Lord is my shepherd. I shall not want. He makes me lie down in green pastures."

sentences = sent_tokenize(text)

print("Sentences:")
for s in sentences:
    print("-", s)
