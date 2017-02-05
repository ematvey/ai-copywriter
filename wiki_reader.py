from wiki import *

import tqdm
import os
import random
import spacy
from multiprocessing import cpu_count
from spacy.tokens import Doc
from textacy.corpora import WikiReader

en = spacy.load('en')
