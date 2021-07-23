import spacy
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

with open("cars.txt") as file:
    dataset = file.read()

nlp = spacy.load("en_core_web_lg")
doc = nlp(dataset)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])