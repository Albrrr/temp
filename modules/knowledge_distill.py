import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from modules.resnet import get_model

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
    if not squared:
        res = res.sqrt()
    return res

def rkd_distance_loss(student_feats, teacher_feats):
    with torch.no_grad():
        t_d = pdist(teacher_feats, squared=False)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = pdist(student_feats, squared=False)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    return F.smooth_l1_loss(d, t_d)

def rkd_angle_loss(student_feats, teacher_feats):
    with torch.no_grad():
        td = (teacher_feats.unsqueeze(0) - teacher_feats.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student_feats.unsqueeze(0) - student_feats.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    return F.smooth_l1_loss(s_angle, t_angle)

class RKDDistiller:
    def __init__(self, teacher_model, device, dist_ratio=25.0, angle_ratio=50.0):
        self.teacher = teacher_model.to(device)
        self.teacher.eval()
        self.device = device

        self.dist_ratio = dist_ratio
        self.angle_ratio = angle_ratio

    def _build_student(self, model_size: str, num_classes: int):
        """Builds a smaller ResNet based on the requested size."""
        return get_model(model_size, num_classes, aux=False).to(self.device)

    def distill(self, model_size: str, dataloader, epochs: int, num_classes: int, lr: float = 0.01, graph_name: str = ""):
        """
        Initializes the student and runs the distillation loop.
        Returns the fully trained student model.
        """
        if not graph_name:
            graph_name = self.__class__.__name__

        student = self._build_student(model_size, num_classes)
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        
        epoch_losses = []

        for epoch in range(epochs):
            student.train()
            running_loss = 0.0

            for images, _, labels, *_ in tqdm(dataloader, desc=f"[{graph_name}] Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device).long()

                with torch.no_grad():
                    t_out = self.teacher(images, return_feat=True)
                    if isinstance(t_out, tuple) and len(t_out) == 3:
                        t_logits, _, t_feats = t_out
                    else:
                        t_logits, t_feats = t_out
                        
                    if t_feats.dim() == 4:
                        t_feats = F.adaptive_avg_pool2d(t_feats, (1, 1)).view(t_feats.size(0), -1)

                s_out = student(images, return_feat=True)
                if isinstance(s_out, tuple) and len(s_out) == 3:
                    s_logits, _, s_feats = s_out
                else:
                    s_logits, s_feats = s_out
                    
                if s_feats.dim() == 4:
                    s_feats = F.adaptive_avg_pool2d(s_feats, (1, 1)).view(s_feats.size(0), -1)

                loss_task = criterion(torch.log(s_logits.clamp(min=1e-8)), labels)
                loss_dist = rkd_distance_loss(s_feats, t_feats)
                loss_angle = rkd_angle_loss(s_feats, t_feats)

                total_loss = loss_task + (self.dist_ratio * loss_dist) + (self.angle_ratio * loss_angle)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

            avg_loss = running_loss/len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        os.makedirs("logs/graphs", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='g', label='Distillation Loss')
        plt.title(f'{graph_name} Distillation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("logs/graphs", f"{graph_name}_distill_loss.png"))
        plt.close()

        return student
    
class OnPolicyRKDDistiller(RKDDistiller):
    def __init__(self, teacher_model, device, dist_ratio=25.0, angle_ratio=50.0, hardness_margin=0.2):
        super().__init__(teacher_model, device, dist_ratio, angle_ratio)
        self.hardness_margin = hardness_margin

    def _on_policy_hard_mining(self, student_feats, teacher_feats):
        with torch.no_grad():
            t_d = pdist(teacher_feats, squared=False)
            t_d = t_d / t_d[t_d > 0].mean()

            s_d = pdist(student_feats, squared=False)
            s_d = s_d / s_d[s_d > 0].mean()

            error_matrix = torch.abs(s_d - t_d)

            hard_mask = error_matrix > self.hardness_margin

            hard_indices = torch.unique(hard_mask.nonzero(as_tuple=False)[:, 0])
            
        return hard_indices

    def distill(self, model_size: str, dataloader, epochs: int, num_classes: int, lr: float = 0.01, graph_name: str = ""):
        if not graph_name:
            graph_name = self.__class__.__name__

        student = self._build_student(model_size, num_classes)
        optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        
        epoch_losses = []

        for epoch in range(epochs):
            student.train()
            running_loss = 0.0

            for images, _, labels, *_ in tqdm(dataloader, desc=f"[On-Policy RKD] Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device).long()

                with torch.no_grad():
                    t_out = self.teacher(images, return_feat=True)
                    if isinstance(t_out, tuple) and len(t_out) == 3:
                        t_logits, _, t_feats = t_out
                    else:
                        t_logits, t_feats = t_out
                        
                    if t_feats.dim() == 4:
                        t_feats = F.adaptive_avg_pool2d(t_feats, (1, 1)).view(t_feats.size(0), -1)

                s_out = student(images, return_feat=True)
                if isinstance(s_out, tuple) and len(s_out) == 3:
                    s_logits, _, s_feats = s_out
                else:
                    s_logits, s_feats = s_out
                    
                if s_feats.dim() == 4:
                    s_feats = F.adaptive_avg_pool2d(s_feats, (1, 1)).view(s_feats.size(0), -1)

                loss_task = criterion(torch.log(s_logits.clamp(min=1e-8)), labels)

                hard_idx = self._on_policy_hard_mining(s_feats, t_feats)

                if len(hard_idx) > 2:
                    s_feats_hard = s_feats[hard_idx]
                    t_feats_hard = t_feats[hard_idx]

                    loss_dist = rkd_distance_loss(s_feats_hard, t_feats_hard)
                    loss_angle = rkd_angle_loss(s_feats_hard, t_feats_hard)
                else:
                    loss_dist = torch.tensor(0.0, device=self.device)
                    loss_angle = torch.tensor(0.0, device=self.device)

                total_loss = loss_task + (self.dist_ratio * loss_dist) + (self.angle_ratio * loss_angle)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

            avg_loss = running_loss/len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        os.makedirs("logs/graphs", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='g', label='Distillation Loss')
        plt.title(f'{graph_name} Distillation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join("logs/graphs", f"{graph_name}_distill_loss.png"))
        plt.close()

        return student