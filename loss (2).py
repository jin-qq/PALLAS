import torch.fft as fft
import torch
import torch.nn as nn
class CC_Loss(nn.Module):
    def __init__(self):
        super(CC_Loss, self).__init__()

    def forward(self, g_s, g_t):
        return torch.tensor(np.average([self.similarity_loss(f_s, f_t).to("cpu").detach().numpy() for f_s, f_t in zip(g_s, g_t)]))

    def similarity_loss(self, f_s, f_t):
        batch = f_s.shape[0]
        f_s = f_s.view(batch, -1)
        f_t = f_t.view(batch, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (batch * batch)
        return loss
class KLDivloss(nn.Module):
    def __init__(self, T):
        super(KLDivloss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
class pos_Loss(nn.Module):
    def __init__(self,beta=2):
        super(pos_Loss, self).__init__()
        self.beta=beta
    def forward(self,s_out,t_out):
        return self.cal_loss(s_out, t_out)
        # return np.average([self.cal_loss(f_s, f_t).to("cpu").detach().numpy() for f_s, f_t in zip(s_out, t_out)])
    def cal_loss(self,s_out,t_out):
        s_out=s_out.flatten(1).to("cuda")
        t_out=t_out.flatten(1).to("cuda")
        s_pos=fk_solver(s_out,False).flatten(1)
        t_pos=fk_solver(t_out,False).flatten(1)
        s_out=s_out.flatten(1).to("cuda")
        t_out=t_out.flatten(1).to("cuda")

        s_fft=fft.fft(F.sigmoid(s_out)).real
        t_fft=fft.fft(F.sigmoid(t_out)).real
        sp_fft=fft.fft(s_pos).real
        tp_fft=fft.fft(t_pos).real
        tp2_fft=tp_fft.view(-1,16,3)
        loss_fn=KLDivloss(3)
        return loss_fn(s_fft,t_fft)*loss_fn(sp_fft,tp_fft)/torch.mean(F.cosine_similarity(s_pos,t_pos)**2)
    
class Diff_Loss(nn.Module):
    def __init__(self):
            super(Diff_Loss).__init__()
    def noise_estimation_loss(model,x0: torch.Tensor,t: torch.LongTensor,e: torch.Tensor,b: torch.Tensor, keepdim=False):
        a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
        output = model(x, t.float())
        if keepdim:
            return (e - output).square().sum(dim=(1, 2, 3)),output
        else:
            return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0),output
    def forward(self,model,x0,t,e,b):
        return noise_estimation_loss(model,x0,t,e,b)