import numpy as np
import tensorflow as tf
from tensorflow import keras
from xml.parsers import expat

from IPython import display

import re

class TextGenerator(keras.callbacks.Callback):
    def __init__(self, model, vocabulary, sequence_length, top_k=1):
        self.__model = model
        self.sequence_length = sequence_length
        self.index_to_word = vocabulary
        self.k = top_k

        self.word_to_index = {}
        for index, word in enumerate(self.index_to_word):
            self.word_to_index[word] = index

    def find_between(self, text, first, last):
      try:
          start = text.index(first) + len(first)
          end = text.index(last, start)
          return text[start:end]
      except ValueError:
          return ''

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype('int32')
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype('float32')
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def tokenize(self, text):
      return [self.word_to_index.get(_, 1) for _ in text.split()]

    @staticmethod
    def display_code(y, canvas_size=(200,200)):
      def convert_to_svg(code):
        return f'''<?xml version="1.0" encoding="UTF-8"?>
      <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{canvas_size[0]}" height="{canvas_size[1]}" viewBox="0 -{canvas_size[1]} {canvas_size[0]} {canvas_size[0]}">
      <rect x="0" y="-{canvas_size[1]}" width="{canvas_size[0]}" height="{canvas_size[1]}" fill="white" stroke-width="2" stroke="black" />
      {code}
      </svg>'''

      def reorder(code):
        elements = re.findall(r'<.+?/>', code)
        orders = [int(re.findall(r'stroke-width *= *"(.+?)"', element)[0]) for element in elements]

        return ''.join([x for _, x in sorted(zip(orders, elements))])

      try:
        ordered_y = reorder(y)
      except ValueError as e:
        print(f'Reorder() exception', e)
        ordered_y = y
      return display.display(display.SVG(convert_to_svg(ordered_y)))

    def display_from_text(self, text):
      try:
        code = self.extract_code(text)
        print('Code: ', code)
        TextGenerator.display_code(code)
      except (ValueError, expat.ExpatError):
        print('Invaid code')

    def revert_separators(self, text):
      separators = [r' \) ', r' \( ']
      replace = [r')', r'(']
      for sep, rep in zip(separators, replace):
        text = re.sub(sep, rep, text)
      
      return text

    def extract_code(self, text):
      return self.revert_separators(self.find_between(text, '$ ', ' %'))

    def generate(self, initial_prompt):
      def wrap_prompt(prompt):
        return re.sub(r',', r' , ', prompt)

      tokens = self.tokenize(re.sub(' +', ' ', f'# {wrap_prompt(initial_prompt)} $'))
      generated_tokens = []
      generated_so_far = 0

      while generated_so_far <= self.sequence_length:
        pad_len = self.sequence_length - len(tokens)
        sample_index = len(tokens) - 1
        if pad_len < 0:
            x = tokens[:self.sequence_length]
            sample_index = self.sequence_length - 1
        elif pad_len > 0:
            x = tokens + [0] * pad_len
        else:
            x = tokens
        
        y, _ = self.__model.predict(np.array([x]))
        sample_token = self.sample_from(y[0][sample_index])
        
        generated_tokens.append(sample_token)
        tokens.append(sample_token)
        generated_so_far = len(generated_tokens)

        if self.detokenize(generated_tokens[-1]) == '%':
          break
        
      txt = ' '.join([self.detokenize(_) for _ in tokens + generated_tokens])
      print(f'Generated text:\n{txt}\n')
      self.display_from_text(txt)