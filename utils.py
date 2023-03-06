import pandas as pd
import numpy as np
import torch
import torch.utils.data
import freqopttest.data as data
import freqopttest.tst as tst
from sklearn.utils import check_random_state
import torch.nn.functional as F


class GenerateData():
    def __init__(self, n_minor, ir):
        self.n_minor = n_minor
        self.ir = ir 
        self.n_major = self.n_minor * self.ir
        self.class0 = np.repeat(0, self.n_major)
        self.class1 = np.repeat(1, self.n_minor)
    
    def _sigma_mx2(self):
        sigma_mx2_standard = np.array([[0.03, 0], [0, 0.03]])
        sigma_mx2 = np.zeros((9, 2, 2))
        for i in range(9):
            sigma_mx2[i, :, :] = sigma_mx2_standard
        if i < 4:
            sigma_mx2[i, 0, 1] = -0.02 - 0.002*i
            sigma_mx2[i, 1, 0] = -0.02 - 0.002*i
        elif i > 4:
            sigma_mx2[i, 0, 1] = -0.02 + 0.002*(i-5)
            sigma_mx2[i, 1, 0] = -0.02 + 0.002*(i-5)
            
        sigma_mx2[4, :, :] = sigma_mx2_standard
        
        return sigma_mx2

    def blob(self, rows=3, cols=3, rs=None):
        """Generate Blob-D for test power"""
        rs = check_random_state(rs)
        mu = np.zeros(2)
        sigma = np.eye(2) * 0.03
        X = rs.multivariate_normal(mu, sigma, size=self.n_major)
        Y = rs.multivariate_normal(mu, np.eye(2), size=self.n_minor)
        
        # Assign to blobs 
        X[:, 0] += rs.randint(rows, size=self.n_major)
        X[:, 1] += rs.randint(cols, size=self.n_major)
        
        Y_row = rs.randint(rows, size=self.n_minor)
        Y_col = rs.randint(cols, size=self.n_minor)
        locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
        for i in range(9):
            corr_sigma = self._sigma_mx2()[i]
            L = np.linalg.cholesky(corr_sigma)
            ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]))
            ind2 = np.concatenate((ind, ind), 1)
            Y = np.where(ind2, np.matmul(Y, L) + locs[i], Y)
        
        return X, Y
    
    def normal_t(self, mu, cov, df, dim):
        if dim > 1:
            x0 = np.random.multivariate_normal(mu, cov, self.n_major)
            x1 = np.column_stack((np.random.standard_t(df, self.n_minor),
                                  np.random.standard_t(df, self.n_minor)))
        else:
            x0 = np.random.normal(mu, cov, self.n_major)
            x1 = np.random.standard_t(df, self.n_minor)
        
        x = np.concatenate((np.column_stack((x0, self.class0)), 
                            np.column_stack((x1, self.class1))))
        np.random.shuffle(x)
        return x
    
    def normal_normal(self, mu0, cov0, mu1, cov1, dim):
        if dim > 1:
            x0 = np.random.multivariate_normal(mu0, cov0, self.n_major)
            x1 = np.random.multivariate_normal(mu1, cov1, self.n_minor)
        else:
            x0 = np.random.normal(mu0, cov0, self.n_major)
            x1 = np.random.normal(mu1, cov1, self.n_minor) 

        x = np.concatenate((np.column_stack((x0, self.class0)), 
                            np.column_stack((x1, self.class1))))
        np.random.shuffle(x)
        return x       
    

def MatConvert(x, device, dtype):
    """convert numpy to a torch tensor"""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def power_stats(tau0, tau1, y_true):
    n0 = len(y_true[y_true == 0])
    n1 = len(y_true[y_true == 1])

    nom = 1 - tau0 - tau1
    denom = np.sqrt((tau0 * (1-tau0)/n0) + (tau1 *(1-tau1)/n1))
    pwr = nom / denom 
    return pwr

def find_optimal_cutoff(output, y_true, cutvals, device):
    accs = []
    pwrs = []
    for i in range(len(cutvals)):
        mask = output[:, 0] > cutvals[i]
        pred = torch.where(mask, torch.tensor(0).to(device), torch.tensor(1).to(device)).long()
        accs.append(torch.mean((pred == y_true).float()).item())
        
        tau0 = torch.mean((pred[y_true == 0] == 1).float()).item()
        tau1 = torch.mean((pred[y_true == 1] == 0).float()).item()
        
        if (tau0 == 0.0) & (tau1 == 1.0):
            pwr = power_stats(tau0, tau1, y_true)
        elif (tau0 == 1.0) & (tau1 == 0.0):
            pwr = power_stats(tau0, tau1, y_true)
        else:
            pwr = power_stats(tau0, tau1, y_true)
        pwrs.append(pwr)
    
    acc_opt_cut = cutvals[accs.index(max(accs))]
    pwr_opt_cut = cutvals[pwrs.index(max(pwrs))]
    
    return acc_opt_cut, pwr_opt_cut
        
    
    
