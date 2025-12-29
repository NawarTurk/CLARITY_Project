MODEL_METADATA = {
    # XLM-R
    "FacebookAI/xlm-roberta-base": {"arch": "xlmr", "lang": "multi", "size": "base"},
    "FacebookAI/xlm-roberta-large": {"arch": "xlmr", "lang": "multi", "size": "large"},
    # DeBERTa / mDeBERTa
    "microsoft/deberta-v3-base": {"arch": "deberta", "lang": "en", "size": "base"},
    "microsoft/deberta-v3-large": {"arch": "deberta", "lang": "en", "size": "large"},
    "microsoft/mdeberta-v3-base": {"arch": "mdeberta", "lang": "multi", "size": "base"},
    # BERT
    "bert-base-uncased": {"arch": "bert", "lang": "en", "size": "base"},
    "bert-base-multilingual-cased": {"arch": "mbert", "lang": "multi", "size": "base"},
    # RoBERTa
    "roberta-base": {"arch": "roberta", "lang": "en", "size": "base"},
    "roberta-large": {"arch": "roberta", "lang": "en", "size": "large"},
}
