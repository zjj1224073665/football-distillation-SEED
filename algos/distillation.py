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
            #range()内有一个参数时，从0开始计数到输入参数；有两个参数时，从参数1计数到参数2；有三个参数时，第三个参数表示步长。
            for mini_batch in data:
                s, _, _, _, _, _, _, _ = mini_batch
                pi, pi_m, v, _ = model(s)
                pi_teacher, pi_m_teacher, v_teacher, _ = sam_model(s)
                #给model模型里传入s和s_prime，模型会输出这几个值
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
                #mean()求均值

                #计算梯度、裁剪梯度、更新网络参数
                model.optimizer.zero_grad()
                #函数会遍历模型的所有参数，通过内置方法截断反向传播的梯度流，再将每个参数的梯度值设为0，即上一次的梯度记录被清空
                loss.backward()
                #损失函数loss是由模型的所有权重w经过一系列运算得到的，若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算
                #在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中
                #如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                #clip_grad_norm_梯度裁剪，对parameters里所有参数的梯度进行规范化，解决的是梯度消失或爆炸的问题，即设定阈值，如果梯度超过阈值，就截断，将梯度变为阈值
                model.optimizer.step()
                #step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值
                #因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度

                tot_loss_lst.append(loss.item())
            last_saved_step = save_model(model, arg_dict, time_steps, last_saved_step)
            loss = np.mean(tot_loss_lst)
            time_steps += arg_dict["batch_size"] * arg_dict["buffer_size"]
            print("step :", time_steps, "loss", loss, "data_q")

