import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import time
import rdflib


def invert(d: dict) -> dict:
    return dict([(v,k) for k,v in d.items()])


def corrupt(mat, n_entities):
    out = mat.copy()
    new_entities = np.random.random_integers(0, n_entities-1, len(mat))
    mask = (np.random.rand(len(mat))+1/2).astype(np.int)
    inv_mask = np.abs(mask-1)
    out[:,0] *= mask
    out[:,2] *= inv_mask
    out[:,0] += inv_mask * new_entities
    out[:,2] += mask * new_entities
    return out

def L1_dist(a, b):
    return tf.reduce_sum(tf.abs(a-b), 1)

def L2_dist(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a-b), 1))


def margin_cost(pos, neg, margin=1.):
    out = margin + pos - neg
    # grad is non-zero only within a certain margin
    in_margin = tf.to_float(tf.greater(out, 0))
    return tf.reduce_sum(out * in_margin)


class TripleArray(object):
    def __init__(self, graph):
        if type(graph) is rdflib.graph.Graph:
            print("%s == %s" % (type(graph), rdflib.graph.Graph))
            self.triples = pd.DataFrame(
                [(str(s), str(p), str(o)) for (s,p,o) in graph if type(o) == rdflib.term.URIRef],
                columns=['s', 'p', 'o']
            )
        elif type(graph) is np.ndarray:
            self.triples = pd.DataFrame(graph, columns=['s', 'p', 'o'])
        elif type(graph) is pd.core.frame.DataFrame:
            self.triples = graph
        else:
            raise TypeError("Unexpected type: %s" % type(graph))

        self.entities            = set(self.triples.s.values) | set(self.triples.o.values)
        self.predicates          = set(self.triples.p.values)
        self.entity_dict    = dict(enumerate(self.entities))
        self.predicate_dict = dict(enumerate(self.predicates))
        self.entity_ids     = invert(self.entity_dict)
        self.predicate_ids  = invert(self.predicate_dict)

        self.arr = np.array([
            self.triples.s.apply(self.entity_ids.get),
            self.triples.p.apply(self.predicate_ids.get),
            self.triples.o.apply(self.entity_ids.get)
        ]).T

    @property
    def n_entities(self):
        return len(self.entity_dict)

    @property
    def n_predicates(self):
        return len(self.predicate_dict)


class TransE(object):
    def __init__(self,
                 dim,
                 optimizer=tf.train.GradientDescentOptimizer(0.01),
                 margin=1.):
        self.dim = dim
        self.optimizer = optimizer
        self.margin = margin

    def _setup_session(self, n_entities, n_predicates):
        '''Builds the TransE computational graph using tensorflow'''
        bound = 6/np.sqrt(self.dim)
        with tf.name_scope('initialize_embeddings') as scope:
            self.entity_embeddings    = tf.Variable(tf.random_uniform([n_entities, self.dim], -bound, +bound),
                                               name='entity_embeddings')
            self.predicate_embeddings = tf.Variable(tf.random_uniform([n_predicates, self.dim], -bound, +bound),
                                               name='predicate_embeddings')

        with tf.name_scope('read_inputs') as scope:
            self.pos_head = tf.placeholder(tf.int32, [None], name='positive_head')
            self.pos_tail = tf.placeholder(tf.int32, [None], name='positive_tail')
            self.neg_head = tf.placeholder(tf.int32, [None], name='corrupted_head')
            self.neg_tail = tf.placeholder(tf.int32, [None], name='corrupted_tail')
            self.link     = tf.placeholder(tf.int32, [None], name='link')

        with tf.name_scope('lookup_embeddings') as scope:
            self.pos_head_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.pos_head)
            self.pos_tail_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.pos_tail)
            self.neg_head_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.neg_head)
            self.neg_tail_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.neg_tail)
            self.link_vec     = tf.nn.embedding_lookup(self.predicate_embeddings, self.link)

        with tf.name_scope('normalize_embeddings') as scope:
            self.pos_head_vec = tf.nn.l2_normalize(self.pos_head_vec, 1)
            self.pos_tail_vec = tf.nn.l2_normalize(self.pos_tail_vec, 1)
            self.neg_head_vec = tf.nn.l2_normalize(self.neg_head_vec, 1)
            self.neg_tail_vec = tf.nn.l2_normalize(self.neg_tail_vec, 1)

        with tf.name_scope('train') as scope:
            # compute loss for true and corrupted triple
            self.pos_dist = L1_dist(tf.add(self.pos_head_vec, self.link_vec), self.pos_tail_vec)
            self.neg_dist = L1_dist(tf.add(self.neg_head_vec, self.link_vec), self.neg_tail_vec)
            diff = self.neg_dist - self.pos_dist
            self.loss = margin_cost(self.pos_dist, self.neg_dist, self.margin)
            self.train = self.optimizer.minimize(self.loss)

        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())

    def _one_rank(self, i):
        s,p,o = self.tarray.arr[i]
        m = self.tarray.n_entities
        dist = self._sess.run(self.pos_dist, feed_dict={self.pos_head: [s],
                                                        self.link: [p],
                                                        self.pos_tail: [o]})

        alt_ids = set(np.arange(m)) - set([i])
        alts = np.array([*alt_ids])
        cdist = self._sess.run(self.pos_dist, feed_dict={self.pos_head: alts,
                                                         self.link: np.ones_like(alts) * p,
                                                         self.pos_tail: np.ones_like(alts) * o})

        return np.sum(dist > cdist)

    def _mean_rank(self, k=10):
        return np.mean([self._one_rank(i) for i in np.random.random_integers(0, self.tarray.n_entities, k)])


    def _rdj_test(self):
        eemb = self._entity_embeddings
        pemb = self._predicate_embeddings

        terminator = eemb.ix['http://data.linkedmdb.org/resource/film/38151'].values
        notebook = eemb.ix['http://data.linkedmdb.org/resource/film/55823'].values

        actor = pemb.ix['http://data.linkedmdb.org/resource/movie/actor'].values
        arnold = eemb.ix['http://data.linkedmdb.org/resource/actor/29369'].values

        return np.sqrt(np.sum(np.square(terminator + actor - arnold))), np.sqrt(np.sum(np.square(notebook + actor - arnold)))

    @property
    def _entity_embeddings(self):
        urns = [self.tarray.entity_dict[i] for i in range(self.tarray.n_entities)]
        data = self._sess.run(self.pos_head_vec, feed_dict={self.pos_head: np.arange(self.tarray.n_entities)})
        return pd.DataFrame(data=data, index=urns)

    @property
    def _predicate_embeddings(self):
        urns = [self.tarray.predicate_dict[i] for i in range(self.tarray.n_predicates)]
        data = self._sess.run(self.link_vec, feed_dict={self.link: np.arange(self.tarray.n_predicates)})
        return pd.DataFrame(data=data, index=urns)

    @property
    def embeddings(self):
        return pd.concat((self._entity_embeddings, self._predicate_embeddings))

    def fit(self, graph, batch_size=1024, num_epochs=10, early_stopping_rounds=5, early_stopping_tolerance=1e-6, warm_start=False):
        if not warm_start:
            if type(graph) in [rdflib.graph.Graph, pd.core.frame.DataFrame]:
                print('Vectorizing graph... ', end='', flush=True)
                self.tarray = TripleArray(graph)
                print('Done.')
            elif type(graph) is TripleArray:
                self.tarray = graph
            else:
                raise TypeError("Unexpected type: %s" % type(graph))

            self.tarray.arr = np.random.permutation(self.tarray.arr)
            self._setup_session(self.tarray.n_entities, self.tarray.n_predicates)

        self.rank_hist = []
        self.loss_hist = []
        n_batches = int(len(self.tarray.arr)/batch_size)
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_losses = []
            for i in range(n_batches):
                sample = self.tarray.arr[i*batch_size:(i+1)*batch_size]
                corrupt_sample = corrupt(sample, self.tarray.n_entities)
                feed  = {self.pos_head: sample[:,0],
                         self.link    : sample[:,1],
                         self.pos_tail: sample[:,2],
                         self.neg_head: corrupt_sample[:,0],
                         self.neg_tail: corrupt_sample[:,2]}

                _, loss = self._sess.run([self.train, self.loss], feed_dict=feed)
                loss /= len(sample)
                epoch_losses.append(loss)

                elapsed = time.time() - epoch_start
                remaining = (n_batches - i) * (elapsed / (1.0 + i))
                print('Batch: {:d}/{:d}, Loss: {:.3f}, ETA: {:.0f} s'.format(i, n_batches, np.mean(epoch_losses), remaining), end='\r')

            self.rank_hist.append(self._mean_rank())
            self.loss_hist.append(np.mean(epoch_losses))
            # print('Epoch {:d} took: {:.0f} s, Loss: {:.3f}, Mean Rank: {:.0f}/{:.0f}'.format(epoch, elapsed, self.loss_hist[-1], self.rank_hist[-1], self.tarray.n_entities))
            print('Epoch {:d} took: {:.0f} s, Loss: {:.3f}'.format(epoch, elapsed, self.loss_hist[-1]))
            # print(self._rdj_test())
            self.last_epoch = epoch
