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

with open("cars.txt") as file:
    dataset = file.readlines()
    for sentence in dataset:
        print("######")
        print("sentence: ", sentence)
        print("######")
        sentence = sentence.lower()
        entities = []
        for word in words:
            word = word.lower()
            if word in sentence:
                start_index = sentence.index(word)
                end_index = len(word) + start_index
                print("word: ", word)
                print("----------------")
                print("start index:", start_index)
                print("end index:", end_index)
                pos = (start_index, end_index, "CAR")
                entities.append(pos)
        element = (sentence.rstrip('\n'), {"entities": entities})

        train_data.append(element)
        print('----------------')
        print("element:", element)

        ("this is my sentence", {"entities": [0, 4, "PREP"]})
        ("this is my sentence", {"entities": [6, 8, "VERB"]})


ner = nlp.get_pipe("ner")

for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])


# Training model
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(60):
        print("Iteration #", iteration)
        # Data shuffle for each iteration
        random.shuffle(train_data)
        losses = {}

        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            for text, annotations in batch:
                # Create an Example object
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], losses=losses, drop=0.1)
                # Update the model
        print("Losses:", losses)


output_dir = Path("/ner/")
nlp.to_disk(output_dir)
print("Saved correctly!")

print("Loading model...")
nlp_updated = spacy.load(output_dir)

#Testing the model:
doc = nlp_updated("Research before you buy or lease a new Tesla vehicle with expert ratings")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents]


doc = nlp_updated("Read the latest Mercedes new car reviews, put through their paces by our team of expert road testers, covering performance, depreciation, servicing cost, ")
print("entities:", [(ent.text, ent.label_) for ent in doc.ents])
