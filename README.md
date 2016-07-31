# graph-embedder
An implementation of the TransE model (https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf) using rdflib and tensorflow.

## Simple Example

```python
graph = rdflib.Graph()
graph = graph.parse('/path/to/input/file')

model = RelationalModel(8)
model.fit(graph)

model.get_embeddings()
```
## TODO

- Make all tensorflow/training parameters configurable (optimizer, learning rate, etc.)
- Add alternate embedding models
- Write tests
- Create embeddings class or utilities module
  - nearest neighbors
  - performance metrics
  - ...
