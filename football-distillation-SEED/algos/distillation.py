import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
import geomloss

def save_model(model, arg_dict, time_steps, last_saved_step):
    if time_steps >= last_saved_step + arg_dict["model_save_interval"]:
        model_dict = {
            'time_steps': time_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
        }
        path = arg_dict["log_dir"]+"/model_"+str(time_steps)+".tar"
        torch.save(model_dict, path)
        print("Model saved :", path)
        return time_steps
    else:
        return last_saved_step

def cost_func(a, b, p=2, metric='cosine'):
    """ a, b in shape: (B, N, D) or (N, D)
    """
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean' and p==2:
        return geomloss.utils.squared_distances(a, b)
    else:
        if a.dim() == 3:
            x_norm = a / a.norm(dim=2)[:, :, None]
            y_norm = b / b.norm(dim=2)[:, :, None]
            M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
        elif a.dim() == 2:
            x_norm = a / a.norm(dim=1)[:, None]
            y_norm = b / b.norm(dim=1)[:, None]
            M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        M = pow(M, p)
        return M

class Algo():
    def __init__(self, arg_dict, device=None):
        self.gamma = arg_dict["gamma"]
        self.K_epoch = arg_dict["k_epoch"]
        self.lmbda = arg_dict["lmbda"]
        self.eps_clip = arg_dict["eps_clip"]
        self.entropy_coef = arg_dict["entropy_coef"]
        self.grad_clip = arg_dict["grad_clip"]


    def train(self, model,sam_model, data,arg_dict, time_steps, last_saved_step):
        tot_loss_lst = []

        while True:
            #range()???????????????????????????0????????????????????????????????????????????????????????????1???????????????2??????????????????????????????????????????????????????
            for mini_batch in data:
                s, _, _, _, _, _, _, _ = mini_batch
                pi, pi_m, v, _ = model(s)
                pi_teacher, pi_m_teacher, v_teacher, _ = sam_model(s)
                #???model???????????????s???s_prime??????????????????????????????
                p = 2
                entreg = .1  # entropy regularization factor for Sinkhorn
                metric = 'cosine'
                OTLoss = geomloss.SamplesLoss(
                    loss='sinkhorn', p=p,
                    cost=lambda a, b: cost_func(a, b, p=p, metric=metric),
                    blur=entreg ** (1 / p), backend='tensorized')
                pi_loss = OTLoss(pi,pi_teacher)
                pi_m_loss = OTLoss(pi_m, pi_m_teacher)
                loss = pi_loss + pi_m_loss
                loss = loss.mean()
                #mean()?????????

                #????????????????????????????????????????????????
                model.optimizer.zero_grad()
                #??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????0???????????????????????????????????????
                loss.backward()
                #????????????loss???????????????????????????w??????????????????????????????????????????w???requires_grads???True??????w??????????????????????????????????????????w??????.grad_fn????????????????????????????????????
                #?????????loss.backward()?????????????????????????????????????????????w??????????????????????????????w???.grad?????????
                #??????????????????tensor.backward()???????????????????????????None?????????loss.backward()?????????optimizer.step()??????
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                #clip_grad_norm_??????????????????parameters?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                model.optimizer.step()
                #step()???????????????????????????????????????????????????????????????????????????????????????
                #??????????????????????????????????????????????????????optimizer.step()?????????????????????loss.backward()?????????????????????

                tot_loss_lst.append(loss.item())
            last_saved_step = save_model(model, arg_dict, time_steps, last_saved_step)
            loss = np.mean(tot_loss_lst)
            time_steps += arg_dict["batch_size"] * arg_dict["buffer_size"]
            print("step :", time_steps, "loss", loss, "data_q")

