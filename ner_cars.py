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

words= [ "Abarth", "Alfa Romeo", "Aston Martin", "Audi", "Bentley", "BMW", "Bugatti", "Cadillac", "Chevrolet", "Chrysler", "Citroën",
        "Dacia","Daewoo","Daihatsu","Dodge","Donkervoort","DS", "Ferrari", "Fiat", "Fisker", "Ford", "Honda", "Hummer", "Hyundai",
        "Infiniti","Iveco","Jaguar","Jeep","Kia","KTM","Lada","Lamborghini","Lancia","Land Rover","Landwind","Lexus","Lotus","Maserati",
        "Maybach","Mazda","McLaren","Mercedes-Benz","MG","Mini","Mitsubishi","Morgan","Nissan","Opel","Peugeot","Porsche","Renault",
        "Rolls-Royce","Saab","Seat","Skoda","Smart","SsangYong","Subaru","Suzuki","Tesla","Toyota","Volkswagen","Volvo",]

train_data = []
