import re
import nltk
import emoji
import numpy as np

class TextToolBox:
    def __init__(self, cbow_model, cbow_skipgram):
        self.stop_words = nltk.corpus.stopwords.words('portuguese') # List of stopwords in portuguese
        self.cbow_model, self.skipgram_model = cbow_model, cbow_skipgram


    def return_model(self, model_name: str) -> any:
        """Return the select model

        Args:
            model_name (str): A name of the model in options menu

        Returns:
            any: Word2Vec model
        """
        return self.cbow_model if model_name =="CBOW" else self.skipgram_model


    def preprocess_text(self, inputString: str) -> list:
        """Preprocessing text to use in NLP aplications

        Args:
            inputString (str): string input

        Returns:
            list: list of tokens
        """
        # Remove caracteres not used in this project
        inputString = self.remove_emojis(inputString)
        inputString = self.remove_punctuation(inputString)
        inputString = self.remove_special_chars(inputString)
        inputString = self.remove_space_unecessary(inputString)

        # Tokenizer and remove portuguese stopwords
        tokens_words = inputString.split(" ")
        tokens = [token for token in tokens_words if token not in self.stop_words]
        return tokens


    def sentence_to_vector(self, sentence: list, model: any) -> float:
        """Transform sentence in a point in n-dimension

        Args:
            sentence (list): List of sentences
            model (any): Model loaded from file

        Returns:
            float: Number represent a vector point
        """
        tokens = self.preprocess_text(sentence) # list of words
        # Filter token present in model list of words
        valid_tokens = [token for token in tokens if token in model.key_to_index]
        
        if len(valid_tokens) > 1:
            # Calcule the mean of vectors points
            vectors = [model[token] for token in valid_tokens]
            return np.sum(vectors, axis=0) / len(valid_tokens)


    def remove_emojis(self, inputString: str) -> str:
        """Remove emojis from string

        Args:
            inputString (str): Phrase with emojis to remove

        Returns:
            str: String without emojis. Return origin string if it not contain any emoji.
        """
        listCaracteres = [char for char in inputString if not emoji.is_emoji(char)]
        return "".join(listCaracteres)


    def remove_punctuation(self, inputString: str) -> str:
        """Removes non-alphabetic characters, preserving Portuguese-specific characters (accents, ç, etc.).

        Args:
            inputString (str): Phrase with punctuation to remove

        Returns:
            str: String without punctuation. Return origin string if it not contain any punctuation.
        """
        return re.sub(r'[^a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕ ]', '', inputString)


    def remove_special_chars(self, inputString: str) -> str:
        """Removes escape characters such as tabs, newlines, carriage returns, form feeds, and vertical tabs from the input string. 

        Args:
            input_string (str): Phrase with special charecteres to remove

        Returns:
            str: String without special charecteres. Return origin string if it not contain any special charecteres.
        """
        return re.sub(r'[\t\n\r\f\v]', ' ', inputString)


    def remove_space_unecessary(self, inputString: str) -> str:
        """Removes unnecessary spaces from a string

        Args:
            input_string (str): Phrase with space unecessary to remove.

        Returns:
            str: String without space unecessary.
        """
        input_string = inputString.strip()
        input_string = re.sub(r'\s+', ' ', input_string)
        input_string = re.sub(r'\s([?.!,:;])', r'\1', input_string)

        return input_string