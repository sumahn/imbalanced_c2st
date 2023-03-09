import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from utils import find_optimal_cutoff, power_stats, PowerLoss
is_cuda = True

# Model
class NeuralNet(torch.nn.Module):
    """define deep neural networks"""
    def __init__(self, x_in, H, x_out, device, dtype):
        super(NeuralNet, self).__init__()
        self._x_in = x_in
        self._H = H 
        self._x_out = x_out
        self.device = device
        self.dtype = dtype
        self._w_c2st = torch.randn([self._x_out, 2]).to(self.device, self.dtype)
        self._b_c2st= torch.randn([1, 2]).to(self.device, self.dtype)
        self._w_c2st.requires_grad = True 
        self._b_c2st.requires_grad = True
        
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True)
        )
    
    def forward(self, input):
        fealant = self.latent(input.to(self.dtype))
        return fealant.to(self.dtype)
    
    def fit(self, dataloader, lr, n_epochs):
        self = self.to(self.device)
        optimizer = torch.optim.Adam(list(self.latent.parameters()) + [self._w_c2st] + [self._b_c2st], lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(n_epochs):
            train_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                y_onehot = F.one_hot(target.view(-1).long(), num_classes=2)
                output_b = self(data.to(self.dtype)).mm(self._w_c2st.to(self.dtype)) + self._b_c2st.to(self.dtype)
                loss = loss_fn(output_b, y_onehot.float())
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        return self
    
    def predict_proba(self, data):
        self.eval()
        f = torch.nn.Sigmoid()
        output = f(self(data.to(self.dtype)).mm(self._w_c2st.to(self.dtype)) + self._b_c2st.to(self.dtype))
        return output.to('cpu').detach().numpy()

        
        
def c2st_nn_fit(S, y, x_in, H, x_out, cutvals, lr, n_epochs, batch_size, device, dtype, validation = False, acc_opt_cut=None, pwr_opt_cut=None):
    """train a deep neural networks for maximizing power"""
    
    if is_cuda:
        model_c2st = LeNet(x_in, H, x_out).cuda
        ()
    else:
        model_c2st = LeNet(x_in, H, x_out)
    
    w_c2st = torch.randn([x_out, 2]).to(device, dtype)
    b_c2st = torch.randn([1, 2]).to(device, dtype)
    w_c2st.requires_grad = True
    b_c2st.requires_grad = True
    
    optimizer = torch.optim.Adam(list(model_c2st.parameters()) + [w_c2st] + [b_c2st], lr=lr)
    loss_fn = torch.nn.BCELoss()
    f = torch.nn.Sigmoid()
        
    # generate train and test indices
    y_np = y.to('cpu').numpy()
    tmp_idx, test_idx, tmp_y, test_y = train_test_split(np.arange(len(y_np)), y_np, test_size=0.5, stratify=y_np)
    
    # generate validation indices 
    train_idx, val_idx, train_y, val_y = train_test_split(tmp_idx, y_np[tmp_idx], test_size=0.5, stratify=y_np[tmp_idx])
    
    if validation == True:
        dataset = torch.utils.data.TensorDataset(S[train_idx], y[train_idx])
    elif validation == False:
        dataset = torch.utils.data.TensorDataset(S[tmp_idx], y[tmp_idx])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)

    for epoch in range(n_epochs+1):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            
            # training model using source data 
            data_source = next(data_iter)
            S_b , y_b = data_source
            y_onehot = F.one_hot(y_b.view(-1).long(), num_classes=2)
            output_b = model_c2st(S_b).mm(w_c2st) + b_c2st
            loss_c2st = loss_fn(f(output_b), y_onehot.float())
            
            optimizer.zero_grad()
            loss_c2st.backward()
            optimizer.step()
            i += 1
        if (validation == True) & (epoch % 100 == 0):
            print("Epochs: {}".format(epoch), end='\t')
            print("Training Binary CE Loss: {}".format(np.round(loss_fn(f(model_c2st(S[train_idx]).mm(w_c2st) + b_c2st), F.one_hot(y[train_idx].view(-1).long(), num_classes=2).float()).item(), 4)), end='\t')
            print("Validation Binary CE Loss: {}".format(np.round(loss_fn(f(model_c2st(S[val_idx]).mm(w_c2st) + b_c2st), F.one_hot(y[val_idx].view(-1).long(), num_classes=2).float()).item(), 4)), end='\t')
            print("Training Accuracy: {}".format(np.round(np.mean(f(model_c2st(S[train_idx]).mm(w_c2st) + b_c2st).max(1, keepdim=True)[1].long().view(-1).detach().cpu().numpy() == y[train_idx].detach().cpu().numpy()), 4)), end='\t')
            print("Validation Accuracy: {}".format(np.round(np.mean(f(model_c2st(S[val_idx]).mm(w_c2st) + b_c2st).max(1, keepdim=True)[1].long().view(-1).detach().cpu().numpy() == y[val_idx].detach().cpu().numpy()), 4)))
            
        elif (validation == False) & (epoch % 100 == 0):
            
            print("Epochs: {}".format(epoch), end='\t')
            print("Training Binary CE Loss: {}".format(np.round(loss_fn(f(model_c2st(S[tmp_idx]).mm(w_c2st) + b_c2st), F.one_hot(y[tmp_idx].view(-1).long(), num_classes=2).float()).item(), 4)), end='\t')
            print("Training Accuracy: {}".format(np.round(np.mean(f(model_c2st(S[tmp_idx]).mm(w_c2st) + b_c2st).max(1, keepdim=True)[1].long().view(-1).detach().cpu().numpy() == y[tmp_idx].detach().cpu().numpy()), 4)))
    # Validation
    if validation ==True:
        model_c2st.eval()
        val_output = f(model_c2st(S[val_idx, :]).mm(w_c2st) + b_c2st)
        acc_opt_cut, pwr_opt_cut = find_optimal_cutoff(val_output, y[val_idx], cutvals, device)
                
        print("acc_opt_cut: ", acc_opt_cut, '\t', 'pwr_opt_cut', pwr_opt_cut)
        return acc_opt_cut, pwr_opt_cut
    
    # Prediction 
    if validation == False:
        model_c2st.eval()
        with torch.no_grad():
            te_output = f(model_c2st(S[test_idx, :]).mm(w_c2st) + b_c2st)
            acc_mask = te_output[:, 0] > acc_opt_cut
            pwr_mask = te_output[:, 0] > pwr_opt_cut

            acc_pred = torch.where(acc_mask, torch.tensor(0).to(device), torch.tensor(1).to(device)).long()
            pwr_pred = torch.where(pwr_mask, torch.tensor(0).to(device), torch.tensor(1).to(device)).long()
            default_pred = te_output.max(1, keepdim=True)[1].long().view(-1)
            
            tau0_a = torch.mean((acc_pred[y[test_idx] == 0] == 1).float()).item()
            tau1_a = torch.mean((acc_pred[y[test_idx] == 1] == 0).float()).item()
            
            tau0_p = torch.mean((pwr_pred[y[test_idx] == 0] == 1).float()).item()
            tau1_p = torch.mean((pwr_pred[y[test_idx] == 1] == 0).float()).item()
            
            tau0_d = torch.mean((default_pred[y[test_idx] == 0] == 1).float()).item()
            tau1_d = torch.mean((default_pred[y[test_idx] == 1] == 0).float()).item()
            
            stat_a = power_stats(tau0_a, tau1_a, y[test_idx])
            stat_p = power_stats(tau0_p, tau1_p, y[test_idx])
            stat_d = power_stats(tau0_d, tau1_d, y[test_idx])
            
            print("Statistics of maximizing acc: {}".format(stat_a), '\t', "tau0_a: {}" .format(tau0_a), '\t', 'tau1_a: {}'.format(tau1_a))
            print("Statistics of default: {}".format(stat_d), '\t', "tau0_d: {}" .format(tau0_d), '\t', 'tau1_d: {}'.format(tau1_d))
            print("Statistics of maximizing pwr: {}".format(stat_p), '\t', "tau0_p: {}" .format(tau0_p), '\t', 'tau1_p: {}'.format(tau1_p))
        
        return stat_a, stat_d, stat_p

def c2st_nn_power(S, y, x_in, H, x_out, learning_rate, n_epochs, batch_size, device, dtype):
    """train a deep network for C2STs"""
    n = S.shape[0]
    if is_cuda:
        model_c2st = LeNet(x_in, H, x_out).cuda()
    else:
        model_c2st = LeNet(x_in, H, x_out)
    
    w_c2st = torch.randn([x_out, 2]).to(device, dtype)
    b_c2st = torch.randn([1, 2]).to(device, dtype)
    w_c2st.requires_grad = True
    b_c2st.requires_grad = True
    
    # generate train and test indices
    y_np = y.to('cpu').numpy()
    train_idx, test_idx, train_y, test_y = train_test_split(np.arange(len(y_np)), y_np, test_size=0.5, stratify=y_np)
    dataset = torch.utils.data.TensorDataset(S[train_idx, :], y[train_idx])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    n0 = len(y[train_idx][y[train_idx] == 0])
    n1 = len(y[train_idx][y[train_idx] == 1])
    criterion = PowerLoss(n0, n1, 1e-10).to(device)
    optimizer = torch.optim.Adam(list(model_c2st.parameters()) + [w_c2st] + [b_c2st], lr = learning_rate)
    f = torch.nn.Sigmoid()
       
    for epoch in range(n_epochs):
    # Train the neural network
    #for epoch in range(n_epoch):
        model_c2st.train()
        i = 0        
        # Forward pass
        while i < len(dataloader):
            # Zero the gradients
            optimizer.zero_grad()
            
            output_b = model_c2st(S[train_idx, :]).mm(w_c2st) + b_c2st
            mask = f(output_b)[:, 0] > 0.5
            y_pred = torch.where(mask, torch.tensor(0).to(device), torch.tensor(1).to(device)).long()
            tau0_b = torch.mean((y_pred[y[train_idx] == 0] == 1).float())
            tau1_b = torch.mean((y_pred[y[train_idx] == 1] == 0).float())
            power = (1 - tau0_b - tau1_b) / torch.sqrt((tau0_b*(1-tau0_b)/len(y[train_idx][y[train_idx] == 0])) + (tau1_b*(1-tau1_b)/len(y[train_idx][y[train_idx] == 1])))
            loss = criterion(output_b, y[train_idx])
            if (tau0_b == 0.0) & (tau1_b == 1.0):
                power= torch.tensor(0.0)
            elif (tau0_b == 1.0) & (tau1_b ==0.0):
                power = torch.tensor(0.0)

            # Backward pass
            loss.backward()
            optimizer.step()
            i += 1
        # Print the loss every 10 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, tau0: {tau0_b.item():.4f}, tau1: {tau1_b.item():.4f}, Asymp Power Loss: {-loss.item():.4f}, Power Stats Training: {power.item():.4f}")
    
    model_c2st.eval()
    with torch.no_grad():
        output = model_c2st(S[test_idx, :]).mm(w_c2st) + b_c2st
        pred = f(output).max(1, keepdim=True)[1]
        tau0 = torch.mean((pred[y[test_idx] == 0] == 1).float())
        tau1 = torch.mean((pred[y[test_idx] == 1] == 0).float())
        stat_c2st = (1 - tau0 - tau1) / torch.sqrt((tau0*(1-tau0)/len(y[test_idx][y[test_idx] == 0])) + (tau1*(1-tau1)/len(y[test_idx][y[test_idx] == 1])))
        # z_alpha = 1.645
        # asymptotic_power = torch.distributions.Normal(0, 1).cdf(z_alpha + stat_c2st)
        print("Power Stats Test: {}".format(stat_c2st), '\t', "tau0_p: {}" .format(tau0), '\t', 'tau1_p: {}'.format(tau1))
    return pred, tau0, tau1, stat_c2st