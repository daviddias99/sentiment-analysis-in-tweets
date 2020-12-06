from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from slang import slangdict
from tqdm import tqdm

preprocessor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
        # terms that will be annotated
        annotate={"elongated", "allcaps", "repeated", 'emphasis', 'censored', 'hashtag'},

        all_caps_tag="wrap",

        fix_text=True, # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter_2018",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter_2018",

        unpack_hashtags=True, # perform word segmentation on hashtags
        unpack_contractions=True, # Unpack contractions (can't -> can not)
        spell_correct_elong=False, # spell correction for elongated words
        spell_correction=True,
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )

def preprocess(name, dataset):
    
    title = "Preprocessing dataset {} ...".format(name)

    return [preprocessor.pre_process_doc(word) for word in tqdm(dataset, title)]