import abc 
from abc import ABC, abstractmethod
import logging  

import pysbd

class TextSplitter(ABC): 
 
    @abstractmethod
    def split_text(self, content: str):
        pass

class SentenceSplitter(TextSplitter):
    def split_text(self, content: str, merge_sentences = 1):
        seg = pysbd.Segmenter(language="en", clean=False)

        content = content.replace("\n", "")

        res = seg.segment(content)

        op = []

        for i in range(0, len(res), merge_sentences):
            seg = "".join(res[ i : (i + merge_sentences)%len(res)])
            op.append(seg)
        return op