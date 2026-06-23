import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset.parser import Parser
from modules.resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, get_model
from modules.margin_trainers import TeZOTrainer, DFATrainer, EndToEndHDTrainer
from modules.trainer import CNNTrainer
from modules.knowledge_distill import RKDDistiller
from modules.hdc_model import HDCModel
from modules.hdc_trainer import HDCTrainer
from modules.ioueval import iouEval
from torchhd import embeddings

TRAIN_E2E = True
TRAIN_STUDENT = False

BATCH_SIZE = 32

HD_DIM = 10000
FEAT_DIM = 128
TEACHER_EPOCHS = 80
DISTILLATION_EPOCHS = 80
HDC_EPOCHS = 10 # still used for final HDC evaluation of the students

# Options for TEACHER_SIZE and STUDENT_SIZE:
# - 'resnet10' : Very small model (often used for student)
# - 'resnet18' : Small model (often used for student)
# - 'resnet34' : Standard model (BasicBlock)
# - 'resnet50' : Larger model (Bottleneck, takes more VRAM)
# - 'resnet101': Much larger model
# - 'resnet152': Extremely large model, ideal for ~90GB VRAM setups

TEACHER_SIZE = "resnet152"
STUDENT_SIZE = "resnet34"

data_root = "data/nuscenes" 

def compute_proto_distances(protos):
    """Computes pairwise cosine distances between class prototypes."""
    norm_protos = F.normalize(protos, dim=1)
    sim_matrix = norm_protos @ norm_protos.t()
    dist_matrix = 1 - sim_matrix
    return dist_matrix

def train_e2e_pipeline(name, model_size, num_classes, loss_weights, device, train_loader, common_rp_weight):
    print(f"\n--- Training {name} Teacher E2E ({model_size}) ---")
    
    model = get_model(model_size, num_classes, aux=True)
    
    log_dir = f"logs/{name.lower()}_teacher_e2e"
    os.makedirs(log_dir, exist_ok=True)
    
    trainer = EndToEndHDTrainer(
        num_classes=num_classes, loss_weights=loss_weights, hd_dim=HD_DIM, feat_dim=FEAT_DIM,
        log_dir=log_dir, device=device, steps_per_epoch=len(train_loader),
        model=model
    )
    trainer.rp_weight = common_rp_weight.to(device)
    
    trainer.train(train_loader, 80) # 80 total epochs
    
    final_dir = f"logs/{name.lower()}_teacher"
    os.makedirs(final_dir, exist_ok=True)
    torch.save({"state_dict": trainer.model.state_dict()}, f"{final_dir}/SENet")
    
    return trainer.model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_cfg = yaml.safe_load(open("config/arch.yaml", 'r'))
    data_cfg = yaml.safe_load(open("config/nuscenes_mini.yaml", 'r'))

    num_classes = len(data_cfg['learning_map_inv'])
    batch_size = arch_cfg['train']['batch_size']

    if not os.path.exists(data_root):
        print(f"WARNING: Data root '{data_root}' not found. Please ensure the dataset is in the correct path.")

    def create_parser(bs):
        return Parser(
            root=data_root,
            train_sequences=data_cfg['split']['train'],
            valid_sequences=data_cfg['split']['valid'],
            labels=data_cfg['labels'],
            color_map=data_cfg['color_map'],
            learning_map=data_cfg['learning_map'],
            learning_map_inv=data_cfg['learning_map_inv'],
            sensor=arch_cfg['dataset']['sensor'],
            max_points=arch_cfg['dataset']['max_points'],
            batch_size=bs,
            workers=arch_cfg['train']['workers']
        )

    parser = create_parser(batch_size)
    parser_fe = create_parser(BATCH_SIZE)
    parser_hdc = create_parser(BATCH_SIZE)
    
    train_loader = parser.get_train_set()
    train_loader_fe = parser_fe.get_train_set()
    train_loader_hdc = parser_hdc.get_train_set()
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

    if TRAIN_E2E:
        teacher_model = train_e2e_pipeline("E2E", TEACHER_SIZE, num_classes, loss_weights, device, train_loader_fe, common_rp_weight)
    else:
        print("\n--- SKIPPED: E2E Teacher Training ---")

    if TRAIN_STUDENT:
        os.makedirs("logs/e2e_student", exist_ok=True)
        os.makedirs("logs/baseline", exist_ok=True)
        
        if not TRAIN_E2E:
            print(f"\n--- Loading E2E Teacher Checkpoint ---")
            teacher_model = get_model(TEACHER_SIZE, num_classes, aux=True)
            ckpt_path = "logs/e2e_teacher/SENet"
            if os.path.exists(ckpt_path):
                teacher_model.load_state_dict(torch.load(ckpt_path)["state_dict"])
            else:
                print(f"[WARNING] Teacher checkpoint not found at {ckpt_path}")
            teacher_model.to(device)

        print(f"\n--- Distilling E2E Teacher -> Student ({STUDENT_SIZE}) ---")
        distiller = RKDDistiller(teacher_model, device)
        e2e_student = distiller.distill(model_size=STUDENT_SIZE, dataloader=train_loader, epochs=DISTILLATION_EPOCHS, num_classes=num_classes, graph_name="E2E")
        torch.save({"state_dict": e2e_student.state_dict()}, "logs/e2e_student/SENet")

        print(f"\n--- Training Baseline Student ({STUDENT_SIZE}) ---")
        baseline_model = get_model(STUDENT_SIZE, num_classes, aux=False)
            
        baseline_trainer = CNNTrainer(num_classes=num_classes, loss_weights=loss_weights, log_dir="logs/baseline", device=device, model=baseline_model, aux_loss=False)
        baseline_trainer.train(train_loader_fe, TEACHER_EPOCHS)
        torch.save({"state_dict": baseline_trainer.model.state_dict()}, "logs/baseline/SENet")
    else:
        print("\n--- SKIPPED: Student Distillation & Baseline Training ---")
        print("\n--- Evaluation SKIPPED: TRAIN_STUDENT is False ---")
        return

    def eval_hdc(path, model_type, name):
        if not os.path.exists(path):
            print(f"[WARNING] Cannot evaluate {name}. Checkpoint not found at {path}")
            return None, None
            
        print(f"\n--- Evaluating HDC Prototypes: {name} ({model_type}) ---")
        hdc_model = HDCModel(num_classes, path, device, HD_DIM, model_type=model_type)
        hdc_model.projection.weight.data = common_rp_weight.clone()
        
        hdc_trainer = HDCTrainer(hdc_model, num_classes, device, retrain_epochs=HDC_EPOCHS)
        hdc_trainer.train(train_loader_hdc)

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

    dist_e2e = compute_proto_distances(protos_e2e) if protos_e2e is not None else None
    dist_base = compute_proto_distances(protos_base) if protos_base is not None else None

    if acc_e2e is not None: results.append(f"{'E2E -> Distill':<25} | {acc_e2e:<10.4f} | {dist_e2e.mean():<15.4f}")
    if acc_base is not None: results.append(f"{'Baseline (Standard)':<25} | {acc_base:<10.4f} | {dist_base.mean():<15.4f}")
    
    results.append("-" * 95)
    results.append("\nPairwise Distance Comparison:")
    header = f"{'Classes':<10} | {'E2E-Dist':<10} | {'Baseline':<10} | {'E-B Diff':<10}"
    results.append(header)
    results.append("-" * len(header))
    
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            de = dist_e2e[i, j].item() if dist_e2e is not None else float('nan')
            db = dist_base[i, j].item() if dist_base is not None else float('nan')
            
            diff = (de - db) if (dist_e2e is not None and dist_base is not None) else float('nan')
            results.append(f"{i:02d} vs {j:02d}:   | {de:<10.4f} | {db:<10.4f} | {diff:<10.4f}")
            
    final_output = "\n".join(results)
    print(final_output)

    with open(os.path.expanduser("experiment_final_ablation.log"), "w") as f:
        f.write(final_output)

if __name__ == "__main__":
    main()