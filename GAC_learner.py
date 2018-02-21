import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Lambda, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.activations import relu, tanh
from keras.constraints import non_neg
from keras.initializers import RandomUniform
from scipy import optimize

#Architechture for policy
hidden_layer_sep_pol = 2
hidden_units_list_pol = [400, 300]

#Architechture for q
hidden_layer_sep_q = 1          # Number of hidden layer for separated s and a braches
hidden_units_list_sep_q = [400] # Number of hidden units in each layer for separate the braches
hidden_layer_merge_q = 1        # Number of hidden layer for the concatenated layer from the branches
hidden_units_list_merge_q = [300]   # Number of hidden units in the concatenated 

initializer = "glorot_uniform"  # Weight initilizer
final_initializer = RandomUniform(minval = -0.003, maxval = 0.003)  # Weight initializer for the final layer
hidden_activation = "relu"

class GAC_learner():
    def __init__(self, ds, da, sess, \
                    epsilon=0.0001, entropy_gamma=0.99, n_taylor=1, min_cov = 0.01, \
                    gamma=0.99, tau = 0.001, lr_q = 0.001, lr_policy = 0.0001, \
                    reg_q = 0, reg_policy = 0, action_bnds = [-1, 1], log_level= 0 ):
        """ Initilize the learner object """
        self.ds = ds
        self.da = da
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_gamma = entropy_gamma
        self.tau = tau
        self.optimizer_Q = Adam(lr=lr_q)
        self.optimizer_policy = Adam(lr=lr_policy)        
        self.reg_q = l2(reg_q )
        self.reg_policy = l2(reg_policy)
        self.entropy = 0        
        self.cov = np.identity(self.da)
        self.n_taylor = n_taylor
        self.log_level = log_level
        self.action_bnds = action_bnds
        
        if action_bnds[0] == -action_bnds[1]:
            self.a_scale = action_bnds[1]
        else:   
            print("Action's upper-bound and lower-bound must be symmetric.") 
            return
        
        if min_cov <= 0:
            min_cov = 0.01

        # Compute a minimum policy's entropy using min_cov    
        self.entropy_0 = 0.5*np.log(min_cov)*self.da + 0.5*self.da*(1+np.log(2*np.pi))      #The minimum entropy                                              
        print("Minimum entropy %f." % (self.entropy_0))

        # Compute the current policy's entropy using current cov
        (sign, logdet) = np.linalg.slogdet(self.cov)
        entropy_init = 0.5*sign*logdet + 0.5*self.da*(1+np.log(2*np.pi))
        self.beta = self.entropy_gamma*(entropy_init - self.entropy_0) + self.entropy_0     #Initial entropy bound
        print("Initial entropy %f from an identity matrix." % entropy_init)

        # Create the actor network, the critic network, and the target critic network.
        self.deep_mean_model = self.create_policy_network()
        self.deep_q_model, self.state_critic, self.action_critic, self.action_grads \
            = self.create_q_network()
        self.target_deep_q_model, _, _, _ = self.create_q_network()
        
        # Attach tf session to keras model and initialize weights
        self.sess = sess
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        # Copy the tnitial weights of the target critic network
        self.target_deep_q_model.set_weights(self.deep_q_model.get_weights())
        
        #self.deep_mean_model.summary()
        #self.deep_q_model.summary()

    def save_model(self, iteration=-1, expname="unknown", model_path = "./Model/"): 
        """ Save models of policy """
        self.deep_mean_model.save(model_path + "%s_mean_I%d.h5" % (expname, iteration)) 
        
    def create_policy_network(self):
        """ Create a policy netwrok """
        input_s = Input(shape=(self.ds,), dtype="float32", name="input_s")    
        
        h_mean = input_s 
        for i in range(0, hidden_layer_sep_pol):
            h_mean = Dense(hidden_units_list_pol[i], activation=hidden_activation, kernel_regularizer=self.reg_policy, kernel_initializer=initializer)(h_mean)

        mean = Dense(self.da, activation="tanh", kernel_regularizer=self.reg_policy, kernel_initializer=final_initializer)(h_mean)
        mean = Lambda(lambda mean, a_scaler: mean*a_scaler, output_shape=(self.da,), arguments={'a_scaler': self.a_scale})(mean)    # This layer scales the tanh output from [-1, 1] to [-a_scale, a_scale]

        deep_mean_model = Model(inputs=[input_s], outputs=[mean])
        deep_mean_model.compile(loss="mse", optimizer=self.optimizer_policy)    # GAC minimizes MSE to learn the policy
        return deep_mean_model

    def create_q_network(self):
        """ Create a q-netwrok """
        input_s = Input(shape=(self.ds,), dtype="float32", name="input_s")  
        input_a = Input(shape=(self.da,), dtype="float32", name="input_a")

        h_s = input_s
        for i in range(0, hidden_layer_sep_q):    # Hidden layers for s branch
            h_s = Dense(hidden_units_list_sep_q[i], activation=hidden_activation, kernel_regularizer=self.reg_q, kernel_initializer=initializer)(h_s)
            
        h_a = input_a
        for i in range(0, hidden_layer_sep_q):    # Hidden layers for a branch
            h_a = Dense(hidden_units_list_sep_q[i], activation=hidden_activation, kernel_regularizer=self.reg_q, kernel_initializer=initializer)(h_a)
            
        h_sa = concatenate([h_s, h_a])  # Concatenation of s branch and a branch

        for i in range(0, hidden_layer_merge_q):    # Another layer with 300 units for the concatenated layer
            h_sa = Dense(hidden_units_list_merge_q[i], activation=hidden_activation, kernel_regularizer=self.reg_q, kernel_initializer=initializer)(h_sa)
            
        output_q = Dense(1, name="output_q", kernel_regularizer=self.reg_q, kernel_initializer=final_initializer)(h_sa)
        deep_q_model = Model(inputs=[input_s, input_a], outputs=[output_q])
        deep_q_model.compile(loss='mse', optimizer=self.optimizer_Q)

        ## action_grads[0] is a [None, da] tensor
        action_grads = tf.gradients(deep_q_model.output, input_a)[0]    # Gradient of Q w.r.t. a

        return  deep_q_model, input_s, input_a, action_grads

    def draw_action(self, s_data):
        """ Sample an action: a ~ pi(a|s; theta) = N(a | mu(s; theta), Cov) """
        mean = self.deep_mean_model.predict_on_batch(np.atleast_2d(s_data.T))
        return np.squeeze(np.random.multivariate_normal(np.reshape(mean.T, (self.da,)), self.cov))
    
    def get_action(self, s_data):
        """ Select an action as the mean of the policy: a = mu(s; theta)) """
        mean = self.deep_mean_model.predict_on_batch(np.atleast_2d(s_data.T))
        return np.reshape(mean.T, (self.da,))  

    def compute_q(self, s_data):
        """ Compute Q-value using the mean action """
        mean = self.deep_mean_model.predict_on_batch(s_data.T)
        q_values = self.deep_q_model.predict_on_batch([s_data.T, mean])
        return np.squeeze(np.mean(q_values, 0))

    def update_q(self, s_data, a_data, sn_data, r_data, d_data):
        """ Update Q-network """
        
        (ds, N) = np.shape(s_data)
        mean_next = self.deep_mean_model.predict_on_batch(sn_data.T)    #[N, da]

        n_sample_q = 10 #number of actions samples to computed the expected future Q-value

        # Compute E_{a'~pi}[Q(s',a')], the expected next state's q-value from policy
        if n_sample_q == 0:
            # if n_sample_q == 0, we use E_{a'~pi}[Q(s',a')] = Q(s', E_{a'~pi}[a'])
            q_values_next = self.target_deep_q_model.predict_on_batch([sn_data.T, mean_next])
        elif n_sample_q == 1:
            chol_cov = np.tile(np.linalg.cholesky(self.cov).T, [N, 1, 1])
            epsilon = np.random.standard_normal((N, self.da))
            an_samples = mean_next + np.einsum('ijk, ij -> ik', chol_cov, epsilon)     # [N, da, da], [N, da] --> [N, da]
            q_values_next = self.target_deep_q_model.predict_on_batch([sn_data.T, an_samples])
        elif n_sample_q > 1:
            an_samples = self.my_multivariate_normal(mean_next, self.cov, n_sample_q)
            q_values_next_tile = self.target_deep_q_model.predict_on_batch([np.repeat(sn_data.T, n_sample_q, 0), an_samples])
            q_values_next =  np.mean(np.reshape(q_values_next_tile, (n_sample_q, N), "F"), 0)
            q_values_next = np.expand_dims(q_values_next, 1)

        d_idx = np.where(1 == d_data[0,:])
        q_values_next[d_idx, 0] = 0         #Q(s',a') = 0 if s' is the terminal state
        y = r_data.T + self.gamma*q_values_next

        # update the Q-network
        loss_q = self.deep_q_model.train_on_batch([s_data.T , a_data.T], y) 

        # update the target Q-networks
        deep_q_model_weights = self.deep_q_model.get_weights()
        target_deep_q_model_weights = self.target_deep_q_model.get_weights()
        for i in range(0, len(deep_q_model_weights)):
            target_deep_q_model_weights[i] = self.tau*deep_q_model_weights[i] + (1-self.tau)*target_deep_q_model_weights[i]
        self.target_deep_q_model.set_weights(target_deep_q_model_weights)
    
        return loss_q 

    def taylor_approximation(self, s_data):
        """ Compute g = grad_q Q(s,a), H = g*g', and H*a, where a is a random sample from pi or just the mean of pi"""
        s_data = s_data.T
        a_for_grad = self.deep_mean_model.predict_on_batch(s_data)     
        if self.n_taylor == 1:
            a_for_grad = self.my_multivariate_normal(a_for_grad, self.cov, 1)
        
        grads = self.sess.run([self.action_grads], feed_dict={self.state_critic: s_data, self.action_critic: a_for_grad})[0]   #size [N, da]
        hessians = -np.einsum('ij, ik -> ijk', grads, grads) # outer-product g*g': [N, da], [N, da] -> [N, da, da]
        hessians_a = np.einsum("ijk, ij -> ik", hessians, a_for_grad)  # H*a: [N, da, da], [N, da] -> [N, da]
            
        return grads, hessians, hessians_a

    def my_multivariate_normal(self, mean, cov, n_sample):
        """ This function generates n_sample from N(mean[n,:], cov) for each n in N 
            :: mean is a [N, da] matrix, cov is da by da matrix
            :: output is [N*n_sample, da], the first n_sample entries are n_sample from N(mean[0,:], cov), and so on.
            This function is supposedly faster than calling np.random.multivariate_normal() N times, once for mean of each state,  
            since here we do cholesky decomposition for the covariance only once.
            """

        (N, da) = np.shape(mean)
        epsilon = np.random.multivariate_normal(np.zeros((da)), np.identity(da), size=(n_sample, N))

        cholesky_tile = np.tile(np.linalg.cholesky(cov).T, (n_sample, N, 1, 1)) # (i k k l) = (n_sample, N, da, da)
        mean_tile = np.tile(mean, (n_sample, 1, 1)) # (i j k) = (n_sample, N, da)
        L_epsilon = np.einsum("ijkl, ijl -> ijk", cholesky_tile, epsilon)
        sample = mean_tile + L_epsilon          # (i j k) = (n_sample, N, da)
        sample = np.transpose(sample, (1,0,2))  # (N, n_sample, da)
        reshape_samples = np.reshape(sample, (n_sample*N, da)) #(N*n_sample, da)

        return reshape_samples

    def update_policy(self, s_data, beta_upd = 0):
        """ Update the actor netwrok by guide actor-critic """
        (ds, N) = np.shape(s_data)
        self.N = N

        # get the policy's mean and compute inverse policy's covariance
        mean = self.deep_mean_model.predict_on_batch([s_data.T])
        cov = np.tile(self.cov, (N, 1, 1))  #[N, da, da] tiled covariance
        self.Q_inv = np.tile(np.linalg.inv(self.cov), (N, 1, 1))    #[N, da, da] tiled inverse covariance
        #self.Q_inv = np.linalg.inv(cov)

        # compute the gradients and Hessian approximations
        q_grads, q_hessian, q_hessian_a0 = self.taylor_approximation(s_data)
        self.W = -q_hessian/2       #**Takes negative half since the code below was coded for Q = -a'*W*a + a'*phi + xi where W is positive definite
        self.L_2nd = q_grads - q_hessian_a0

        """ This part is for optimizing the dual function for eta and omega. We can skip this part if eta and omega are pre-specified """
        # pre-computed these variables to avoid repeated computation in the dual_function
        self.L_1st = np.linalg.solve(cov, mean)  #Q_inv * mean    #this is more numerically stable than using the inverse
        self.phiQinvphi = np.sum(np.einsum("ij, ij -> i", mean, self.L_1st))/N
        self.WQ2 = 2*np.einsum("ijk, ikl -> ijl", self.W, cov)
        sign, logdet = np.linalg.slogdet(cov)
        self.logdet2piQ = np.sum(sign*logdet)/N + self.da*np.log(2*np.pi)        #0.5 factor is there in the dual evaluation
           
        # minimize the dual function to find the dual variables eta and omega.
        dual_var = np.array([0.05, 0.05]) #initial eta and omega     
        res = optimize.minimize(self.dual_function, dual_var, method = "SLSQP", bounds=((1e-10, 1e6), (1e-10, 1e6)) , jac=True, options={"maxiter": 500})    
        eta, omega = res.x[0], res.x[1]
        """"""

        # Compute the guide mean, which is a result of a 2nd order update with curvature F = eta*self.Q_inv + 2*self.W  
        F_Ndd = eta*self.Q_inv + 2*self.W                   
        guide_mean = mean + np.linalg.solve(F_Ndd, q_grads)    
        
        # Compute the guide covariance by averaging the guide covariance
        F_inv_Ndd = np.linalg.inv(F_Ndd) #N D D
        guide_cov = (eta+omega)*F_inv_Ndd
        self.cov = np.sum(guide_cov, 0)/N

        # Update the policy network by minizing mse using Adam (1 gradient step)
        loss_pol = self.deep_mean_model.train_on_batch(s_data.T, guide_mean)         

        # Commpute a new entropy bound for the next iteration
        #guide_cov_F = F_Ndd/(eta+omega) #inverse covariance
        (sign, logdet) = np.linalg.slogdet(guide_cov)
        logdetNewQ = np.sum(sign*logdet)
        self.entropy = 0.5*logdetNewQ/N + 0.5*self.da*(1+np.log(2*np.pi))
        if beta_upd:
            (sign, logdet) = np.linalg.slogdet(guide_cov)
            logdetNewQ = np.sum(sign*logdet)
            self.entropy = 0.5*logdetNewQ/N + 0.5*self.da*(1+np.log(2*np.pi))
            beta_tmp = self.entropy_gamma*(self.entropy - self.entropy_0) + self.entropy_0
            self.beta = max(beta_tmp, self.entropy_0)

        # prepare logging variables if needed
        if self.log_level >= 2:
            if self.da == 1:
                self.sumQ = self.cov
            else:
                self.sumQ = np.diagonal(self.cov)

            if self.log_level >= 3:
                if self.da == 1:
                    self.minW = np.amin(self.W, 0)
                    self.maxW = np.amax(self.W, 0)
                    self.minQ = np.amin(guide_cov, 0)
                    self.maxQ = np.amax(guide_cov, 0)
                else:
                    self.minW = np.amin(np.diagonal(self.W, 0, 1, 2), 0)
                    self.maxW = np.amax(np.diagonal(self.W, 0, 1, 2), 0)
                    self.minQ = np.amin(np.diagonal(guide_cov, 0, 1, 2), 0)
                    self.maxQ = np.amax(np.diagonal(guide_cov, 0, 1, 2), 0)
                
        return loss_pol, eta, omega

    def dual_function(self, dual_var):
        eta = dual_var[0]
        omega = dual_var[1]

        F_Ndd = eta*self.Q_inv + 2*self.W #(da da N)
        L_Nd = eta*self.L_1st + self.L_2nd #(da N)
        #F_chol = np.linalg.cholesky(F_Ndd)
        
        # Theoretically, F should always be positive definite. 
        # However, F may be semi-definite numerically due to underflow eigenvalues. 
        # This is likely to happen when eta is too small and eigenvalues of Hessians and covariance are also small in magnitude.
        try:
            F_chol = np.linalg.cholesky(F_Ndd)
        except:
            return (100000,  np.array([100000, 100000]))


        logdetFinv = -2*np.sum(np.log(np.diagonal(F_chol, 0, 1, 2)))  #cholesky version, mostly faster than slogdet
        logdet2pietaFinv = logdetFinv/self.N + self.da*np.log(2*np.pi) + self.da*np.log(eta+omega)

        F_inv_L = np.linalg.solve(F_Ndd, L_Nd)  # N da        
        L_F_inv_L = np.sum(np.einsum("ij, ij-> i", L_Nd, F_inv_L))/self.N

        # compute the dual
        first_term = eta*self.epsilon - omega*self.beta - eta*self.logdet2piQ/2 + (eta+omega)*logdet2pietaFinv/2
        second_term = L_F_inv_L/2 - eta*self.phiQinvphi/2
        dual = first_term + second_term

        # We use Q_inv*F_inv = (F*Q)_inv = ((eta*Q_inv + 2W)*Q)_inv = (eta*I + 2*W*Q)_inv. This avoids another product operation.
        Q_inv_F_inv = np.linalg.inv(eta*np.eye(self.da) + self.WQ2)

        # compute the gradient w.r.t. eta and omega
        grad_quad1 = 2*np.sum(np.einsum("ij, ij->i", self.L_1st, F_inv_L)) 
        grad_quad2 = np.sum(np.einsum("ij, ijl, il-> i", F_inv_L, Q_inv_F_inv, L_Nd))    
        grad_const = self.epsilon - ( self.logdet2piQ - logdet2pietaFinv + (eta+omega)*np.trace(np.sum(Q_inv_F_inv, 0))/self.N - self.da)/2   #this term is correct
        grad_eta = grad_const + (grad_quad1 - grad_quad2)/self.N/2 - self.phiQinvphi/2
        grad_omega = -self.beta + self.da/2 + logdet2pietaFinv/2
        grad = np.array([grad_eta, grad_omega])   
        
        return (dual, grad)
