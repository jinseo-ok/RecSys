import numpy as np


# Equation of FM model
def inference(data):
    num_data = len(data)
    scores = np.zeros(num_data)
    for n in range(num_data):
        feat_idx = data[n][0]
        val = np.array(data[n][1])
        
        # linear feature score
        linear_feature_score = np.sum(w[feat_idx] * val)
        
        # factorized feature score
        vx = v[feat_idx] * (val.reshape(-1, 1))
        cross_sum = np.sum(vx, axis=0)
        square_sum = np.sum(vx*vx, axis=0)
        cross_feature_score = 0.5 * np.sum(np.square(cross_sum) - square_sum)
        
        # Model's equation
        scores[n] = b + linear_feature_score + cross_feature_score

    # Sigmoid transformation for binary classification
    scores = 1.0 / (1.0 + np.exp(-scores))
    return scores

def _get_loss(self, y_data, y_hat):
    """
    Calculate loss with L2 regularization (two type of coeficient - w,v)
    """
    l2_norm = 0
    if self._l2_reg:
        w_norm = np.sqrt(np.sum(np.square(self.w)))
        v_norm = np.sqrt(np.sum(np.square(self.v)))
        l2_norm = self._l2_lambda * (w_norm + v_norm)
    return -1 * np.sum( (y_data * np.log(y_hat)) + ((1 - y_data) * np.log(1 - y_hat)) ) + l2_norm
    
def _stochastic_gradient_descent(self, x_data, y_data):
    """
    Update each coefs (w, v) by Gradient Descent
    """
    for data, y in zip(x_data, y_data):
        feat_idx = data[0]
        val = data[1]
        vx = self.v[feat_idx] * (val.reshape(-1, 1))

        # linear feature score
        linear_feature_score = np.sum(self.w[feat_idx] * val)

        # factorized feature score
        vx = self.v[feat_idx] * (val.reshape(-1, 1))
        cross_sum = np.sum(vx, axis=0)
        square_sum = np.sum(vx * vx, axis=0)
        cross_feature_score = 0.5 * np.sum(np.square(cross_sum) - square_sum)

        # Model's equation
        score = linear_feature_score + cross_feature_score
        y_hat = 1.0 / (1.0 + np.exp(-score))
        cost = y_hat - y

        if self._l2_reg:
            self.w[feat_idx] = self.w[feat_idx] - cost * self._lr * (val + self._l2_lambda * self.w[feat_idx])
            self.v[feat_idx] = self.v[feat_idx] - cost * self._lr * ((sum(vx) * (val.reshape(-1, 1)) - (vx * (val.reshape(-1, 1)))) + self._l2_lambda * self.v[feat_idx])
        else:
            self.w[feat_idx] = self.w[feat_idx] - cost * self._lr * val
            self.v[feat_idx] = self.v[feat_idx] - cost * self._lr * (sum(vx) * (val.reshape(-1, 1)) - (vx * (val.reshape(-1, 1))))