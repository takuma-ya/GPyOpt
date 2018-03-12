# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import EvaluatorBase
import scipy
from ...util.general import samples_multidimensional_uniform
import numpy as np

class SynchronousTS(EvaluatorBase):
    """
    Class for the batch method on 'Batch Bayesian optimization via local penalization' (Gonzalez et al., 2016). 

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.
    :normalize_Y: whether to normalize the outputs.

    """
    def __init__(self, acquisition, batch_size):
        super(SynchronousTS, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size

    def compute_batch(self, duplicate_manager=None,context_manager=None):
        """
        Computes the elements of the batch sequentially by penalizing the acquisition.
        """
        from ...acquisitions import AcquisitionTS
        assert isinstance(self.acquisition, AcquisitionTS)
        
        X_batch,_ = self.acquisition.optimize()
        k=1
        
        # --- GET the remaining elements
        while k<self.batch_size:
            new_sample,_ = self.acquisition.optimize()
            X_batch = np.vstack((X_batch,new_sample))
            k +=1
       
        return X_batch



