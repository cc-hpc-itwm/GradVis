import torch
import tensorflow as tf

class Base_NNModel():
    """
    Provides an interface to the NN model.
    """
    def __init__(self):
        pass


    def get_parameters(self, filename=None):
        """
        Get the model parameters.
        If filename is provided, it will return the model parameters of the checkpoint. The model is not changed.
        If no arguments are given, it will return the current model parameters.
        Returns a dictionary, comprising of parameter identifiers as keys and numpy arrays as data containers.
        Weights and biases are supposed to have "weight" or "bias" appear in their ID string!

        Args:
            filename: string of the checkpoint, of which the parameters should be returned.
        Return:
            python dictionary of string parameter IDs mapped to numpy arrays.
        """
        raise NotImplementedError("Override this function!")


    def set_parameters(self, parameter_dict):
        """
        Set the model parameters.
        The input dictionary must fit the model parameters!

        Args:
            parameter_dict: python dictionary, mapping parameter id strings to numpy array parameter values.
        """
        raise NotImplementedError("Override this function!")


    def calc_loss(self):
        """
        Calculates the loss of the NN.

        Return:
            The loss, based on the parameters loaded into the model.
        """
        raise NotImplementedError("Override this function!")


class PyTorch_NNModel(Base_NNModel):
    """
    Provides an interface to the PyTorch NN model.
    """
    def __init__(self, model, trigger_fn, filename):
        super(PyTorch_NNModel, self).__init__()
        #self.parameter = self._torch_params_to_numpy(torch.load(filename))
        self.model = model
        self.parameter = self.get_parameters(filename)
        self.trigger_fn = trigger_fn
        self.set_parameters(self.parameter)


    def get_parameters(self, filename=None):
        if filename is None:
            return self.parameter
        else:
            return self._torch_params_to_numpy(torch.load(filename))
            #self.model.load_state_dict(torch.load(filename))
            #print(self.model)
            #return self._torch_params_to_numpy(dict(self.model.named_parameters()))


    def get_param_vec(self):
        return np.concatenate( [ar.flatten() for ar in list(parameter.values())], axis=None)


    def set_parameters(self, parameter_dict):
        #if self.parameter.dims != parameter.dims or any(self.parameter.shape != parameter.shape):
        #    raise RuntimeError("New parameter shape is not the same as old one!")
        self.model.load_state_dict(self._numpy_params_to_torch(parameter_dict))


    def calc_loss(self):
        return self.trigger_fn()


    def _numpy_params_to_torch(self, parameter):
        new_param = dict()
        for key,val in parameter.items():
            new_param[key] = torch.tensor(val)
        return new_param


    def _torch_params_to_numpy(self, parameter):
        new_param = dict()
        for key,val in parameter.items():
            new_param[key] = val.cpu().detach().numpy()
        return new_param


class Tensorflow_NNModel(Base_NNModel):
    """
    Provides an interface to the Tensorflow NN model.
    """
    def __init__(self, model, trigger_fn, filename, number_of_steps=2):
        print("Build Tensorflow Model...")
        super(Tensorflow_NNModel, self).__init__()
        print("Making Saver...")
        self.saver = tf.train.Saver() #Saver used to restore parameters from file

        # Create Session
        hooks=[] #optinal hooks
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session_creator = tf.train.ChiefSessionCreator(checkpoint_filename_with_path=filename, config=config)
        print("Making Session...")
        self.mon_sess = tf.train.MonitoredSession(session_creator=session_creator, hooks=hooks)

        self.number_of_steps=number_of_steps #maximum number of iteration steps per evaluation
        self.model = model
        self.total_loss = trigger_fn
        print("Initializing Parameters...")
        self.parameter = self._tf_params_to_numpy()
        #TODO: make dict of numpy arrays from self.parameter
        print("Done.")


    def get_parameters(self, filename=None):
        if filename is None:
            return self.parameter
        else:
            self.saver.restore(self.mon_sess, filename)
            tmp_params = self._tf_params_to_numpy()
            # restore old state, since getter is not changing model
            self.set_parameters(self.parameter)
            self.parameter = tmp_params
            return self.parameter


    def set_parameters(self, parameter_dict):
        for var in tf.trainable_variables():
            var.load(parameter_dict[var.name[:-2]], self.mon_sess)


    def calc_loss(self):
        average_loss=0
        for i in range(self.number_of_steps):
            current_loss = self.mon_sess.run(self.total_loss)
            #print(np.argmax(label,axis=1))
            average_loss += current_loss
        average_loss /= self.number_of_steps
        print("Average Loss: "+str(average_loss))
        return average_loss


    def _tf_params_to_numpy(self):
        new_param = dict()
        for var in tf.trainable_variables():
            new_param[var.name[:-2]] = var.eval(self.mon_sess)
        return new_param

