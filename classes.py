import numpy as np
import theano as theano
import theano.tensor as T
import operator
from collections import defaultdict, OrderedDict

class RNN:
    """
    Single layer RNN class
    
    x (len(sent),word_dim): sentence with words in idx vector
    U (hidden_dim,word_dim): embedding vector, which transform x into embedded vector
                             with dim=hidden_dim
    W (hidden_dim,hidden_dim): propogate last hidden output to current input
    s_t (hidden_dim): output/state for hidden node
    
    """
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # lecun_uniform: Uniform initialization scaled by the square root of the number of inputs
        U = np.random.uniform(low=-np.sqrt(1./word_dim),
                              high=np.sqrt(1./word_dim),
                              size=(hidden_dim, word_dim))
        V = np.random.uniform(low=-np.sqrt(1./hidden_dim),
                              high=np.sqrt(1./hidden_dim),
                              size=(word_dim, hidden_dim))
        W = np.random.uniform(low=-np.sqrt(1./hidden_dim),
                              high=np.sqrt(1./hidden_dim),
                              size=(hidden_dim, hidden_dim))
        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t] # need to take [0] as nnet.softmax returns a 2D tensor
        
        # scan arg: sequences (x_t) | prior outputs (s_t_prev) | non-sequences (U, V, W)
        [o,s], updates = theano.scan(
            fn=forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))], # pass only s_t
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
        
        # Assign functions
        self.pred_prob = theano.function([x], o)
        self.pred_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dU, dV, dW])
        
        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x,y,learning_rate], [], 
                                        updates=[(self.U, self.U - learning_rate * dU),
                                                 (self.V, self.V - learning_rate * dV),
                                                 (self.W, self.W - learning_rate * dW)])
    
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)
    
    # def errors(self, y):
        
class GRU:
    """
    A 2-layer GRU class
    """
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./word_dim),
                               np.sqrt(1./word_dim),
                               (hidden_dim, word_dim)) # (output_dim, input_dim)
        U = np.random.uniform(-np.sqrt(1./hidden_dim),
                               np.sqrt(1./hidden_dim),
                               (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim),
                               np.sqrt(1./hidden_dim),
                               (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim),
                               np.sqrt(1./hidden_dim),
                               (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)
        
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        # parameters of the model
        self.params = [self.E, self.U, self.W, self.V, self.b, self.c]
        
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
        
        x = T.ivector('x')
        y = T.ivector('y')
        
        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e = E[:,x_t]
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0]) # update gate
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1]) # remember gate
            # Now, some memory (s_t1_prev, remember percentage is controlled by r_t1) is retained
            # it is mashed into current input to form the current output c_t1
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2]) 
            # internal state is only partially updated, this is controlled by the update gate z_t1
            s_t1 = z_t1 * c_t1 + (T.ones_like(z_t1) - z_t1) * s_t1_prev
            
            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = z_t2 * c_t2 + (T.ones_like(z_t2) - z_t2) * s_t2_prev
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            return [o_t, s_t1, s_t2]
        [o, s, s2], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        
        # Assign functions
        self.pred_prob = theano.function([x], o)
        self.pred_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

        grad_updates = sgd_updates_adadelta(self.params, cost, word_vec_name='E')
        self.sgd_step_adadelta = theano.function([x, y], [], updates=grad_updates)
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)


def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


# def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
#     # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
#     model.bptt_truncate = 1000
#     # Calculate the gradients using backprop
#     bptt_gradients = model.bptt(x, y)
#     # List of all parameters we want to chec.
#     model_parameters = ['U', 'V', 'W']
#     # Gradient check for each parameter
#     for pidx, pname in enumerate(model_parameters):
#         # Get the actual parameter value from the mode, e.g. model.W
#         parameter_T = operator.attrgetter(pname)(model)
#         parameter = parameter_T.get_value()
#         print("Performing gradient check for parameter %s with size %d." % (
#             pname, np.prod(parameter.shape)))
#         # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
#         it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
#         while not it.finished:
#             ix = it.multi_index
#             # Save the original value so we can reset it later
#             original_value = parameter[ix]
#             # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
#             parameter[ix] = original_value + h
#             parameter_T.set_value(parameter)
#             gradplus = model.calculate_total_loss([x],[y])
#             parameter[ix] = original_value - h
#             parameter_T.set_value(parameter)
#             gradminus = model.calculate_total_loss([x],[y])
#             estimated_gradient = (gradplus - gradminus)/(2*h)
#             parameter[ix] = original_value
#             parameter_T.set_value(parameter)
#             # The gradient for this parameter calculated using backpropagation
#             backprop_gradient = bptt_gradients[pidx][ix]
#             # calculate The relative error: (|x - y|/(|x| + |y|))
#             numerator = np.abs(backprop_gradient - estimated_gradient)
#             denominator = np.abs(backprop_gradient) + np.abs(estimated_gradient)
#             relative_error = numerator / denominator
#             # relative_error = np.abs(backprop_gradient - estimated_gradient)/
#             # (np.abs(backprop_gradient) + np.abs(estimated_gradient))
            
#             # If the error is to large fail the gradient check
#             if relative_error > error_threshold:
#                 print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
#                 print("+h Loss: %f" % gradplus)
#                 print("-h Loss: %f" % gradminus)
#                 print("Estimated_gradient: %f" % estimated_gradient)
#                 print("Backpropagation gradient: %f" % backprop_gradient)
#                 print("Relative Error: %f" % relative_error)
#                 return 
#             it.iternext()
#         print("Gradient check for parameter %s passed." % (pname))
