#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.
For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy


import pt_core_news_sm

# Import aux functions
from aux_timer import *


dirpath = "/home/contato/train-nlp"
outpath = "/home/contato/train-nlp/model"

# training data
TRAIN_DATA = list()
path2 = "/home/contato/train-nlp/selected_v1.txt"
#with open(dirpath+"wiki_07_d_entities_PERSON_6_NORP_2_FAC_0_ORG_0_GPE_8_LOC_0_PRODUCT_0_EVENT_0_WORK_OF_ART_0_LAW_0_LANGUAGE_0_DATE_1_TIME_0_PERCENT_1_MONEY_0_QUANTITY_0_ORDINAL_1_CARDINAL_4.txt") as f:

with open(path2) as f:
    for line in f:
        #a = json.loads('{"url": "https://en.wikipedia.org/wiki?curid=12", "text": "Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions."}')
        TRAIN_DATA.append(eval(line.strip('\n')))  

'''
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]
'''
'''
TRAIN_DATA = [('A cena do hip hop holandês é dividida pelas cidades maiores da Holanda.', {'entities': [(18, 26, 'NORP')]}),
('do 1998', {'entities': [(3, 7, 'DATE')]}),
('Em 18 de outubro de 2012, a equipe anunciou que', {'entities': [(3, 24, 'DATE')]}),
('e invernos suaves com', {'entities': [(2, 10, 'DATE')]}),
('a cada trinta minutos, bem como serviços mais freqüentes', {'entities': [(0, 21, 'TIME')]}),
('Abril de 2018, o Roma foi derrotado por 4-1 para o Barcelona', {'entities': [(0, 13, 'DATE'), (17, 21, 'GPE'), (40, 41, 'CARDINAL'), (51, 60, 'GPE')]}),
('de Arrhenius e Brønsted', {'entities': [(3, 12, 'PERSON'), (15, 23, 'PERSON')]}),
('manteiga custa US $ 10 a libra, bacon US $ 5 a libra, farinha US $ 3 a libra e um alqueire', {'entities': [(20, 22, 'MONEY'), (43, 44, 'MONEY'), (67, 68, 'MONEY')]}),
('A tripulação da Apollo 10 também são os humanos', {'entities': [(23, 25, 'CARDINAL')]}),
('o primeiro', {'entities': [(2, 10, 'ORDINAL')]}),
]
'''

@timing
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model='pt_core_news_sm', output_dir=outpath, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == '__main__':
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]