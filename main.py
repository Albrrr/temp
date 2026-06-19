import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset.parser import Parser
from modules.resnet import ResNet10, ResNet18, ResNet34
from modules.separation_trainers import TeZOTrainer, DFATrainer
from modules.trainer import CNNTrainer
from modules.knowledge_distill import RKDDistiller
from modules.hdc_model import HDCModel
from modules.hdc_trainer import HDCTrainer
from modules.ioueval import iouEval
from torchhd import embeddings

TRAIN_TEZO = True
TRAIN_DFA = True
TRAIN_BASELINE = True

HD_DIM = 10000
FEAT_DIM = 128
TEACHER_FEATURE_EXTRACTOR_EPOCHS = 80
STUDENT_FEATURE_EXTRACTOR_EPOCHS = 80
HDC_RETRAIN_EPOCHS = 10
STUDENT_SIZE = "resnet10"

data_root = "data/nuscenes" 

def compute_proto_distances(protos):
    """Computes pairwise cosine distances between class prototypes."""
    norm_protos = F.normalize(protos, dim=1)
    sim_matrix = norm_protos @ norm_protos.t()
    dist_matrix = 1 - sim_matrix
    return dist_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_cfg = yaml.safe_load(open("config/arch.yaml", 'r'))
    data_cfg = yaml.safe_load(open("config/nuscenes_mini.yaml", 'r'))

    num_classes = len(data_cfg['learning_map_inv'])
    batch_size = arch_cfg['train']['batch_size']

    if not os.path.exists(data_root):
        print(f"WARNING: Data root '{data_root}' not found. Please ensure the dataset is in the correct path.")

    parser = Parser(
        root=data_root,
        train_sequences=data_cfg['split']['train'],
        valid_sequences=data_cfg['split']['valid'],
        labels=data_cfg['labels'],
        color_map=data_cfg['color_map'],
        learning_map=data_cfg['learning_map'],
        learning_map_inv=data_cfg['learning_map_inv'],
        sensor=arch_cfg['dataset']['sensor'],
        max_points=arch_cfg['dataset']['max_points'],
        batch_size=batch_size,
        workers=arch_cfg['train']['workers']
    )
    
    train_loader = parser.get_train_set()
    val_loader = parser.get_valid_set()

    content = torch.zeros(num_classes)
    for k, v in data_cfg['content'].items():
        if k in data_cfg['learning_map']:
            content[data_cfg['learning_map'][k]] += v
    loss_weights = 1.0 / (content + arch_cfg['train']['epsilon_w'])
    loss_weights[0] = 0
    ignore_idx = (loss_weights < 1e-10).nonzero(as_tuple=True)[0].tolist()

    print("\n[INFO] Generating common random HD projection...")
    torch.manual_seed(42)  # Ensure consistency across runs for RP matrix
    _proj_emb = embeddings.Projection(FEAT_DIM, HD_DIM)
    common_rp_weight = _proj_emb.weight.detach().to(device)
    
    random_protos = torch.randn(num_classes, HD_DIM, device=device)
    random_protos = F.normalize(random_protos, dim=1)

    os.makedirs("logs/tezo_student", exist_ok=True)
    os.makedirs("logs/dfa_student", exist_ok=True)
    os.makedirs("logs/baseline", exist_ok=True)

    if TRAIN_TEZO:
        print("\n--- [Branch 1] Training TeZO Teacher (ResNet34) ---")
        tezo_trainer = TeZOTrainer(
            num_classes=num_classes, loss_weights=loss_weights, hd_dim=HD_DIM, feat_dim=FEAT_DIM,
            log_dir="logs/tezo_teacher", device=device, steps_per_epoch=len(train_loader)
        )
        tezo_trainer.rp_weight = common_rp_weight
        tezo_trainer.set_class_protos(random_protos)
        tezo_trainer.train(train_loader, TEACHER_FEATURE_EXTRACTOR_EPOCHS)

        print(f"\n--- [Branch 1] Distilling TeZO Teacher -> Student ({STUDENT_SIZE}) ---")
        distiller_tezo = RKDDistiller(tezo_trainer.model, device)
        tezo_student = distiller_tezo.distill(model_size=STUDENT_SIZE, dataloader=train_loader, epochs=STUDENT_FEATURE_EXTRACTOR_EPOCHS, num_classes=num_classes, graph_name="TeZO")
        torch.save({"state_dict": tezo_student.state_dict()}, "logs/tezo_student/SENet")
    else:
        print("\n--- [Branch 1] SKIPPED: TeZO Training & Distillation ---")

    if TRAIN_DFA:
        print("\n--- [Branch 2] Training DFA Teacher (ResNet34) ---")
        dfa_trainer = DFATrainer(
            num_classes=num_classes, loss_weights=loss_weights, hd_dim=HD_DIM, feat_dim=FEAT_DIM,
            log_dir="logs/dfa_teacher", device=device, steps_per_epoch=len(train_loader)
        )
        dfa_trainer.rp_weight = common_rp_weight
        dfa_trainer.set_class_protos(random_protos)
        dfa_trainer.train(train_loader, TEACHER_FEATURE_EXTRACTOR_EPOCHS)

        print(f"\n--- [Branch 2] Distilling DFA Teacher -> Student ({STUDENT_SIZE}) ---")
        distiller_dfa = RKDDistiller(dfa_trainer.model, device)
        dfa_student = distiller_dfa.distill(model_size=STUDENT_SIZE, dataloader=train_loader, epochs=STUDENT_FEATURE_EXTRACTOR_EPOCHS, num_classes=num_classes, graph_name="DFA")
        torch.save({"state_dict": dfa_student.state_dict()}, "logs/dfa_student/SENet")
    else:
        print("\n--- [Branch 2] SKIPPED: DFA Training & Distillation ---")

    if TRAIN_BASELINE:
        print(f"\n--- [Branch 3] Training Baseline Student ({STUDENT_SIZE}) ---")
        if STUDENT_SIZE == "resnet10":
            baseline_model = ResNet10(num_classes)
        else:
            baseline_model = ResNet18(num_classes)
            
        baseline_trainer = CNNTrainer(num_classes=num_classes, loss_weights=loss_weights, log_dir="logs/baseline", device=device, model=baseline_model, aux_loss=False)
        baseline_trainer.train(train_loader, TEACHER_FEATURE_EXTRACTOR_EPOCHS)

        torch.save({"state_dict": baseline_trainer.model.state_dict()}, "logs/baseline/SENet")
    else:
        print("\n--- [Branch 3] SKIPPED: Baseline Training ---")

    def eval_hdc(path, model_type, name):
        if not os.path.exists(path):
            print(f"[WARNING] Cannot evaluate {name}. Checkpoint not found at {path}")
            return None, None
            
        print(f"\n--- Evaluating HDC Prototypes: {name} ({model_type}) ---")
        hdc_model = HDCModel(num_classes, path, device, HD_DIM, model_type=model_type)
        hdc_model.projection.weight.data = common_rp_weight.clone()
        
        hdc_trainer = HDCTrainer(hdc_model, num_classes, device, retrain_epochs=HDC_RETRAIN_EPOCHS)
        hdc_trainer.train(train_loader)

        independent_evaluator = iouEval(num_classes, device, ignore_idx)
        acc = hdc_trainer.validate(val_loader, independent_evaluator)
        
        hdc_model.sync_class_weights()
        protos = hdc_model.classify.weight.data.clone()
        return acc, protos

    acc_tezo, protos_tezo = eval_hdc("logs/tezo_student/SENet", STUDENT_SIZE, "TeZO-Distilled")
    acc_dfa, protos_dfa = eval_hdc("logs/dfa_student/SENet", STUDENT_SIZE, "DFA-Distilled")
    acc_base, protos_base = eval_hdc("logs/baseline/SENet", STUDENT_SIZE, "Baseline")

    results = []
    results.append("\n" + "="*95)
    results.append("FINAL ABLATION STUDY: GENERALIZATION & PROTOTYPE SEPARATION")
    results.append("="*95)
    results.append(f"{'Model Strategy':<25} | {'Val mIoU':<10} | {'Avg Proto Dist':<15}")
    results.append("-" * 95)

    dist_tezo = compute_proto_distances(protos_tezo) if protos_tezo is not None else None
    dist_dfa = compute_proto_distances(protos_dfa) if protos_dfa is not None else None
    dist_base = compute_proto_distances(protos_base) if protos_base is not None else None

    if acc_tezo is not None: results.append(f"{'TeZO -> Distill':<25} | {acc_tezo:<10.4f} | {dist_tezo.mean():<15.4f}")
    if acc_dfa is not None:  results.append(f"{'DFA -> Distill':<25} | {acc_dfa:<10.4f} | {dist_dfa.mean():<15.4f}")
    if acc_base is not None: results.append(f"{'Baseline (Standard)':<25} | {acc_base:<10.4f} | {dist_base.mean():<15.4f}")
    
    results.append("-" * 95)
    results.append("\nPairwise Distance Comparison:")
    header = f"{'Classes':<10} | {'TeZO-Dist':<10} | {'DFA-Dist':<10} | {'Baseline':<10} | {'T-B Diff':<10}"
    results.append(header)
    results.append("-" * len(header))
    
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dt = dist_tezo[i, j].item() if dist_tezo is not None else float('nan')
            dd = dist_dfa[i, j].item() if dist_dfa is not None else float('nan')
            db = dist_base[i, j].item() if dist_base is not None else float('nan')
            
            diff = (dt - db) if (dist_tezo is not None and dist_base is not None) else float('nan')
            results.append(f"{i:02d} vs {j:02d}:   | {dt:<10.4f} | {dd:<10.4f} | {db:<10.4f} | {diff:<10.4f}")
            
    final_output = "\n".join(results)
    print(final_output)

    with open(os.path.expanduser("experiment_final_ablation.log"), "w") as f:
        f.write(final_output)

if __name__ == "__main__":
    main()