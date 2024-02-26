# Context Aware Chunker
When performing semantic search using vector similarity, one of the key issues that arises is the size of the chunk you are using.

The size of the chunk affects a lot of things, including the accuracy of your result, the amount of contextual information retained at inference time, and accuracy of retrieval.

One of the easiest ways to boost accuracy is to retain highly correlated information in a single atomic chunk as opposed to creating multiple, since this might be missed when performing semantic search. 

## How does this package work?
The idea is quite simple. Language models are extremely good at knowing when two pieces of text belong together.

When they do, the perplexity remains low, but when they aren't, the perplexity is much higher. 

Based on this, we can merge two groups of text together, creating the perfect chunk of highly correlated information

## Usage

> WARNING: Please note that this is an alpha release and is only suitable for testing, not for production

### Installation
```
pip install context_aware_chunking
```
### Python Code
```python
text = "<INSERT TEXT HERE>"

from context_aware_chunker.chunking_models import T5ChunkerModel
from context_aware_chunker.text_splitter import SentenceSplitter

#This module will help you in finding relevant sentences from unstructured text
splitter = SentenceSplitter()

'''
Responsible for determining which sentence segments to merge or separate
If you have more GPU power you can try using larger models
'''
chunking_agent = T5ChunkerModel('t5-small')

'''
Here, merge_sentences decides how many sentences will be in one split of the sentences
Default is 1, you can increase and see
'''
split_content = splitter.split_text(text, merge_sentences = 1)

chunks = chunking_agent.chunk(split_content)

for chunk in chunks:
  print(chunk)
```
