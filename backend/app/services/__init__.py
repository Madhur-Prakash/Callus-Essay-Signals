"""Detection pipeline services.

Layering (each layer only depends on the ones above it):

``nlp``                  spaCy singleton, segmentation, token views
``stylometry``           surface + lexical + syntactic measurements
``burstiness``           sentence-rhythm statistics
``repetition``           n-gram and syntactic-template repetition
``probability_analyzer`` causal-LM token log-probabilities (the "instrument")
``corpus_analyzer``      similarity to the human / AI reference corpora
``style_shift``          within-document deviation and change points
``feature_extractor``    assembles the sentence- and document-level vectors
``classifier``           the trained models (our own decision maker)
``calibration``          probability calibration + confidence bands
``explanation_engine``   feature values -> deterministic evidence statements
``detector``             orchestration: text in, explained verdict out
"""
