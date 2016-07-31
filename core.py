import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import time
import rdflib
import re
import pickle

def invert(d: dict) -> dict:
    return dict([(v,k) for k,v in d.items()])

class TripleArray(object):
    def __init__(self, graph):
        self.triples = pd.DataFrame(
            [(str(s), str(p), str(o)) for (s,p,o) in graph if type(o) == rdflib.term.URIRef],
            columns=['s', 'p', 'o']
        )

        entities            = set(self.triples.s.values) | set(self.triples.o.values)
        predicates          = set(self.triples.p.values)
        self.entity_dict    = dict(enumerate(entities))
        self.predicate_dict = dict(enumerate(predicates))
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

def L2_dist(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a-b)))

def margin_cost(pos, neg, margin=1.):
    out = margin + pos - neg
    # grad is non-zero only within a certain margin
    in_margin = tf.to_float(tf.greater(out, 0))
    return tf.reduce_sum(out * in_margin)

class RelationalModel(object):
    def __init__(self, dim):
        self.dim = dim

    def _setup_session(self, n_entities, n_predicates):
        bound = 6/np.sqrt(self.dim)
        with tf.name_scope('initialize_embeddings') as scope:
            self.entity_embeddings    = tf.Variable(tf.random_uniform([n_entities, self.dim],-bound, +bound),
                                                    name='entity_embeddings')
            self.predicate_embeddings = tf.Variable(tf.random_uniform([n_predicates, self.dim],-bound, +bound),
                                                    name='predicate_embeddings')

        with tf.name_scope('read_inputs') as scope:
            self.pos_head = tf.placeholder(tf.int32, [None], name='positive_head')
            self.pos_tail = tf.placeholder(tf.int32, [None], name='positive_tail')
            self.neg_head = tf.placeholder(tf.int32, [None], name='corrupted_head')
            self.neg_tail = tf.placeholder(tf.int32, [None], name='corrupted_tail')
            self.link     = tf.placeholder(tf.int32, [None], name='link')

        with tf.name_scope('lookup_embeddings') as scope:
            self.pos_head_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.pos_head)
            pos_tail_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.pos_tail)
            neg_head_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.neg_head)
            neg_tail_vec = tf.nn.embedding_lookup(self.entity_embeddings, self.neg_tail)
            self.link_vec     = tf.nn.embedding_lookup(self.predicate_embeddings, self.link)

        with tf.name_scope('normalize_embeddings') as scope:
            pos_head_vec = tf.nn.l2_normalize(self.pos_head_vec, 1)
            pos_tail_vec = tf.nn.l2_normalize(pos_tail_vec, 1)
            neg_head_vec = tf.nn.l2_normalize(neg_head_vec, 1)
            neg_tail_vec = tf.nn.l2_normalize(neg_tail_vec, 1)

        with tf.name_scope('train') as scope:
            # compute loss for true and corrupted triple
            pos_dist = L2_dist(tf.add(self.pos_head_vec, self.link_vec), pos_tail_vec)
            neg_dist = L2_dist(tf.add(neg_head_vec, self.link_vec), neg_tail_vec)
            diff = neg_dist - pos_dist
            loss = margin_cost(pos_dist, neg_dist, np.sqrt(self.dim))
#             self.train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#             self.train = tf.train.AdamOptimizer(epsilon=1e-15).minimize(loss)
            self.train = tf.train.AdadeltaOptimizer(0.01).minimize(loss)


        self._sess = tf.Session()
        init = tf.initialize_all_variables()
        self._sess.run(init)

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
    def get_embeddings(self):
        return pd.concat((self._entity_embeddings, self._predicate_embeddings))

    def fit(self, graph, batch_size=1024, num_epochs=10):
        print('Vectorizing graph...', end='')
        sys.stdout.flush()
        self.tarray = TripleArray(graph)
        print('Done')
        self._setup_session(self.tarray.n_entities, self.tarray.n_predicates)
        self.rank_hist = []
        n_batches = int(len(self.tarray.arr)/batch_size)
        for epoch in range(num_epochs):
            epoch_start = time.time()
            for i in range(n_batches):
                sample = self.tarray.arr[i*batch_size:(i+1)*batch_size]
                corrupt_sample = corrupt(sample, self.tarray.n_entities)
                feed  = {self.pos_head: sample[:,0],
                         self.link    : sample[:,1],
                         self.pos_tail: sample[:,2],
                         self.neg_head: corrupt_sample[:,0],
                         self.neg_tail: corrupt_sample[:,2]}

                self._sess.run(self.train, feed_dict=feed)

                elapsed = time.time() - epoch_start
                remaining = (n_batches - i) * (elapsed / (1.0 + i))
                print('Batch: {:d}/{:d}, ETA: {:.0f} s'.format(i, n_batches, remaining), end='\r')

            self.rank_hist.append(self.mean_rank())
            print('Epoch {:d} took: {:.0f} s, Mean Rank: {:.3f}'.format(epoch, elapsed, self.rank_hist[-1]))
            self.last_epoch = epoch

    def mean_rank(self):
        arr = self.tarray.arr
        carr = corrupt(arr, self.tarray.n_entities)

        semb = self._sess.run(self.pos_head_vec, feed_dict={self.pos_head: arr[:,0]})
        pemb = self._sess.run(self.link_vec,     feed_dict={self.link: arr[:,1]})
        oemb = self._sess.run(self.pos_head_vec, feed_dict={self.pos_head: arr[:,2]})

        csemb = self._sess.run(self.pos_head_vec, feed_dict={self.pos_head: carr[:,0]})
        cpemb = self._sess.run(self.link_vec,     feed_dict={self.link: carr[:,1]})
        coemb = self._sess.run(self.pos_head_vec, feed_dict={self.pos_head: carr[:,2]})

        dists = np.sqrt(np.sum(np.square((semb + pemb) - oemb), axis=1))
        cdists = np.sqrt(np.sum(np.square((csemb + cpemb) - coemb), axis=1))

        return (cdists > dists.mean()).sum() / len(cdists)