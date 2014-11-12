from helpers import *

class VariationalAutoencoder:

    def __init__( self , n_x = 784 , n_z = 50 , hidden_q = (500 , 500) , hidden_p = (500 , 500)  , learning_rate = 0.001, momentum = 0.9 , batch_size = 32  , data_source = "../data/mnist.pkl.gz" , seed = 323):

        # Load the dataset
        f = gzip.open('../data/mnist.pkl.gz', 'rb')
        self.train_set, self.valid_set, self.test_set = cPickle.load(f)
        f.close()
        self.shared_X = theano.shared(numpy.asarray(self.train_set[0], dtype=theano.config.floatX), borrow=True)
        self.shared_valid_X= theano.shared(numpy.asarray(self.valid_set[0], dtype=theano.config.floatX), borrow=True)

        #Learning Parameters.
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

        #Helper Parameters.
        self.n_x = n_x
        self.n_z = n_z
        self.theano_rng = RandomStreams(numpy.random.RandomState(seed).randint(2 ** 30))
        
        ################################
        # Initialise Q parameters (fi) #
        ################################
        self.q_h_W = []
        self.q_h_b = []
        self.q_h = []
        self.x = T.matrix('x', dtype=theano.config.floatX)
        self.e_noise = self.theano_rng.normal(size=(self.batch_size , self.n_z), avg=0.0, std=1.0)

        #Iterate over hidden q layers and create shared variables for weights and biases 
        #and symbolic variables for the layer calculations.
        previous_layer_size = n_x
        for layer_index , layer_size in enumerate(hidden_q):
            self.q_h_W.append(theano.shared(value=initialW( previous_layer_size , layer_size ), name='q_W_h_' + str(layer_index), borrow=True))
            self.q_h_b.append(theano.shared(value=numpy.zeros(layer_size, dtype=theano.config.floatX), name='q_b_h_' + str(layer_index), borrow=True))
            if layer_index == 0:
                self.q_h.append(sigmoid(self.x ,self.q_h_W[layer_index] ,self.q_h_b[layer_index]))
            else:
                self.q_h.append(sigmoid(self.q_h[layer_index-1] ,self.q_h_W[layer_index] ,self.q_h_b[layer_index]))
            previous_layer_size = layer_size

        #Create extra layers of shared and symbolic variables for the factors Mu and Sigma of Q(z|x).
        self.q_Z_X_mu_W = theano.shared(value=initialW( previous_layer_size ,  n_z ), name='q_W_mu', borrow=True)
        self.q_Z_X_mu_b = theano.shared(value=numpy.zeros(n_z, dtype=theano.config.floatX), name='q_b_mu', borrow=True)
        self.q_Z_X_sigma_W = theano.shared(value=initialW( previous_layer_size ,  n_z ), name='q_W_sigma', borrow=True)    
        self.q_Z_X_sigma_b = theano.shared(value=numpy.zeros(n_z, dtype=theano.config.floatX), name='q_b_sigma', borrow=True)
        self.q_Z_X_mu =  T.dot( self.q_h[len(self.q_h)-1] , self.q_Z_X_mu_W ) + self.q_Z_X_mu_b
        self.q_Z_X_log_sigma = T.dot( self.q_h[len(self.q_h)-1] , self.q_Z_X_sigma_W ) + self.q_Z_X_sigma_b
        
        #Q outputs
        self.z = self.q_Z_X_mu + T.exp(self.q_Z_X_log_sigma)*self.e_noise
        self.NKLD = 0.5 * T.sum( 1 + 2*self.q_Z_X_log_sigma - self.q_Z_X_mu**2  - T.exp(2*self.q_Z_X_log_sigma) )
        
        ###################################
        # Initialise P parameters (theta) #
        ###################################
        self.p_h_W = []
        self.p_h_b = []
        self.p_h = []

        #Iterate over hidden p layers and create shared variables for weights and biases 
        #and symbolic variables for the layer calculations.
        previous_layer_size = n_z
        for layer_index , layer_size in enumerate(hidden_p):
            self.p_h_W.append(theano.shared(value=initialW( previous_layer_size , layer_size ), name='p_W_h_' + str(layer_index), borrow=True))
            self.p_h_b.append(theano.shared(value=numpy.zeros(layer_size, dtype=theano.config.floatX), name='p_b_h_' + str(layer_index), borrow=True))
            if layer_index == 0:
                self.p_h.append(sigmoid(self.z ,self.p_h_W[layer_index] ,self.p_h_b[layer_index]))
            else:
                self.p_h.append(sigmoid(self.p_h[layer_index-1] ,self.p_h_W[layer_index] ,self.p_h_b[layer_index]))
            previous_layer_size = layer_size

        #Create extra layers of shared and symbolic variables for the factors Mu and Sigma of P(x|z).
        self.p_X_Z_mu_W = theano.shared(value=initialW( previous_layer_size ,  n_x ), name='p_W_mu', borrow=True)
        self.p_X_Z_mu_b = theano.shared(value=numpy.zeros(n_x, dtype=theano.config.floatX), name='p_b_mu', borrow=True)
        self.p_X_Z_sigma_W = theano.shared(value=initialW( previous_layer_size ,  n_x ), name='p_W_sigma', borrow=True)    
        self.p_X_Z_sigma_b = theano.shared(value=numpy.zeros(n_x, dtype=theano.config.floatX), name='p_b_sigma', borrow=True)
        self.p_X_Z_mu =  T.dot( self.p_h[len(self.p_h)-1] , self.p_X_Z_mu_W  ) + self.p_X_Z_mu_b
        self.p_X_Z_log_sigma = T.dot( self.p_h[len(self.p_h)-1] , self.p_X_Z_sigma_W ) + self.p_X_Z_sigma_b
        
        #P outputs
        self.log_p_X_Z = T.sum(-(0.5 * numpy.log(2 * numpy.pi) + self.p_X_Z_log_sigma) - 0.5 * ((self.x - self.p_X_Z_mu) / T.exp(self.p_X_Z_log_sigma))**2)
        
       
        ##########################
        # Training Configuration #
        ##########################
        #Objective Function
        self.L = self.NKLD + self.log_p_X_Z

        #Gather all parameters.
        self.parameters = self.q_h_W + self.q_h_b + self.p_h_W + self.p_h_b 
        self.parameters += [ self.q_Z_X_mu_W , self.q_Z_X_mu_b , self.q_Z_X_sigma_W , self.q_Z_X_sigma_b ]
        self.parameters += [ self.p_X_Z_mu_W , self.p_X_Z_mu_b , self.p_X_Z_sigma_W , self.p_X_Z_sigma_b ]

        #get gradients.
        self.gradients = T.grad(self.L , self.parameters)

        #Create rmsprop updates.
        self.updates = []
        for param , grad in zip(self.parameters, self.gradients):
            rms_ = theano.shared(numpy.zeros_like(param.get_value()), name=param.name + '_rms')
            rms = self.momentum * rms_ + (1 - self.momentum) * grad * grad
            self.updates.append( ( rms_, rms) )
            self.updates.append( ( param, param + self.learning_rate * grad / T.sqrt(rms + 1e-8) ) )

        #Training Function.
        self.index = T.lscalar('index')
        self.fn_train = theano.function( 
	                        inputs =  [ self.index ] ,
	                        outputs = [ self.L ],
	                        updates = self.updates,
	                        givens = { self.x : self.shared_X[self.index*self.batch_size:(self.index+1)*batch_size] }
	                    )

        #Calculate Lower Bound on Validation Set.
        self.fn_validation_get_lower_bound = theano.function([self.index] , self.L , givens = { self.x : self.shared_valid_X[self.index*self.batch_size:(self.index+1)*batch_size] } )


    def train(self, max_epochs = 10000000 , max_n_of_batches = 100000000 ):
        epoch = 0
        LB = []
        while(epoch < max_epochs):
            LB.append(numpy.mean([self.fn_train(i) for i in range(0 , min(max_n_of_batches , int(self.train_set[0].shape[0]/self.batch_size) ) )]))
            if not epoch % 10:
                validation_lb = numpy.mean( [ self.fn_validation_get_lower_bound(i) for i in range(0,int(self.valid_set[0].shape[0]/self.batch_size)) ] )
                print "Epoch: " , epoch ,  " , LB: " , LB[-1] , " , Validation LB: " , validation_lb
                print "---------------------------------------------------------------------"
            epoch += 1
