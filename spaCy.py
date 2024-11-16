import spacy

def split_sentences_spacy(text):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

text = "But if he is in a position to control the federal budget, what better way to deal with an investigation than to just defund the agency that is supposed to be doing that investigation?"
sentences = split_sentences_spacy(text)
print(sentences)