from Th_VAE import *

#Initialise
vae = VariationalAutoencoder(
            n_z = 3,
            hidden_q=(100,),
            hidden_p = (100,),
            learning_rate = 0.001,
            momentum = 0.9,
            batch_size = 32,
            data_source = "../data/mnist.pkl.gz",
            seed = 3223)   

#Train
vae.train( max_epochs = 100 , max_n_of_batches = 100 )