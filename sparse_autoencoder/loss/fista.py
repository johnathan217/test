import math

import torch

def FISTA(x, M, alpha, num_iter, C=None):
    """FISTA algorithm for sparse coding.

    Args:
        x: Input data. [batch_size, activation_size]
        M: Dictionary / Decoder weights. [activation_size, n_feats]
        alpha: Regularization parameter.
        num_iter: Number of iterations.
    """

    # get estimate of Lipschitz constant
    L = torch.max(torch.linalg.eigh(torch.mm(M, M.t()), UPLO='L')[0])
    stepsize = 1. / L

    batch_size = x.size(0)
    n_feats = M.size(1)

    tk_n = 1.
    tk = 1.
    if C is None:
        C = torch.cuda.FloatTensor(batch_size, n_feats).fill_(0)

    for t in range(num_iter):
        C_pre = C

        # # gradient step on smooth part
        x_hat = torch.mm(C, M.t())
        res = x - x_hat
        grad = -2 * torch.mm(res, M)
        C = C.add(-stepsize * grad)

        # soft thresholding step on non-smooth part
        C = C.sub(stepsize * alpha).clamp(min=0.)

        # calculate FISTA momentum term
        tk = tk_n
        tk_n = (1+math.sqrt(1+4*tk**2))/2

        # apply FISTA momentum term
        C = C.add(C.sub(C_pre).mul((tk-1)/(tk_n)))

    res = x - torch.mm(C, M.t())
    return C, res
