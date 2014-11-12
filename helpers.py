import cPickle, gzip, os, sys, numpy, time,theano, PIL.Image
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from numpy.linalg import *

#Helper Functions
def initialW(n_in , n_out):
    return numpy.asarray(
                numpy.random.RandomState(123).uniform(
                    low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                    high=4 * numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ), dtype=theano.config.floatX
            )

def noisy(input, corruption_level, theano_rng):
    return theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX) * input

def sigmoid(input , W ,  b):
    return  T.nnet.sigmoid(T.dot( input , W ) + b)   

def softplus(input , W ,  b):
    return T.log(T.exp(T.dot( input , W ) + b) + 1)

def relu(input , W ,  b):
    y = T.maximum(0.0, T.dot(input, W) + b)
    return(y)

def dropout(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output

def whiten(X): 
    Xcov = numpy.cov(X) 
    d,V = eigh(Xcov) 
    D = diag(1./sqrt(d+5.0)) 
    W = dot(dot(V,D),V.T) 
    Xw = dot(W,X) 
    return Xw
