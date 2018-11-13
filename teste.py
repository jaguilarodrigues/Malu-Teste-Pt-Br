import spacy
nlp = spacy.load('pt')
doc = nlp(u'Você encontrou o livro que eu te falei, Carla?')
print(doc.text.split())
print([token for token in doc])
print([token.orth_ for token in doc if not token.is_punct])

tokens = [token for token in doc]

print(tokens[0].similarity(tokens[5]))

print(tokens[0].similarity(tokens[3]))
