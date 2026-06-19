import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from dataset.parser import Parser
from modules.resnet import ResNet10, ResNet18
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

NUSCENES_DIR = "data/nuscenes_raw"
CONVERTED_DIR = "data/nuscenes"
LOG_DIR = "logs"
CNN_CHECKPOINT = os.path.join(LOG_DIR, "SENet")
HDC_CHECKPOINT = os.path.join(LOG_DIR, "hdc.pth")
data_root = CONVERTED_DIR


def convert_dataset():
    from dataset.export_semantickitti import KittiConverter
    if not os.path.isdir(NUSCENES_DIR):
        raise FileNotFoundError(f"Raw NuScenes data not found at '{NUSCENES_DIR}'")
    print(f"Converting {NUSCENES_DIR} -> {CONVERTED_DIR}")
    converter = KittiConverter(nusc_dir=NUSCENES_DIR, nusc_skitti_dir=CONVERTED_DIR)
    converter.nuscenes_gt_to_semantickitti()


def load_context():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_cfg = yaml.safe_load(open("config/arch.yaml", "r"))
    data_cfg = yaml.safe_load(open("config/nuscenes_mini.yaml", "r"))

    num_classes = len(data_cfg["learning_map_inv"])
    batch_size = arch_cfg["train"]["batch_size"]

    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"Converted dataset not found at '{data_root}'. Run: python main.py --convert"
        )

    parser = Parser(
        root=data_root,
        train_sequences=data_cfg["split"]["train"],
        valid_sequences=data_cfg["split"]["valid"],
        labels=data_cfg["labels"],
        color_map=data_cfg["color_map"],
        learning_map=data_cfg["learning_map"],
        learning_map_inv=data_cfg["learning_map_inv"],
        sensor=arch_cfg["dataset"]["sensor"],
        max_points=arch_cfg["dataset"]["max_points"],
        batch_size=batch_size,
        workers=arch_cfg["train"]["workers"],
    )

    train_loader = parser.get_train_set()
    val_loader = parser.get_valid_set()

    content = torch.zeros(num_classes)
    for k, v in data_cfg["content"].items():
        if k in data_cfg["learning_map"]:
            content[data_cfg["learning_map"][k]] += v
    loss_weights = 1.0 / (content + arch_cfg["train"]["epsilon_w"])
    loss_weights[0] = 0
    ignore_idx = (loss_weights < 1e-10).nonzero(as_tuple=True)[0].tolist()

    return {
        "device": device,
        "arch_cfg": arch_cfg,
        "num_classes": num_classes,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "loss_weights": loss_weights,
        "ignore_idx": ignore_idx,
    }


def build_student_model(num_classes: int):
    if STUDENT_SIZE == "resnet10":
        return ResNet10(num_classes)
    return ResNet18(num_classes)


def train_cnn():
    ctx = load_context()
    device = ctx["device"]
    num_classes = ctx["num_classes"]
    epochs = ctx["arch_cfg"]["train"]["max_epochs"]

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n--- Training CNN feature extractor ({STUDENT_SIZE}, {epochs} epochs) ---")
    model = build_student_model(num_classes)
    trainer = CNNTrainer(
        num_classes=num_classes,
        loss_weights=ctx["loss_weights"],
        log_dir=LOG_DIR,
        device=device,
        model=model,
        aux_loss=False,
    )
    trainer.train(ctx["train_loader"], epochs)

    net = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    torch.save({"state_dict": net.state_dict()}, CNN_CHECKPOINT)
    print(f"Saved CNN checkpoint to {CNN_CHECKPOINT}")


def train_hdc():
    ctx = load_context()
    device = ctx["device"]

    if not os.path.exists(CNN_CHECKPOINT):
        raise FileNotFoundError(
            f"CNN checkpoint not found at '{CNN_CHECKPOINT}'. Run: python main.py --train-cnn"
        )

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n--- Training HDC classifier on frozen CNN ({STUDENT_SIZE}) ---")
    torch.manual_seed(42)
    rp_weight = embeddings.Projection(FEAT_DIM, HD_DIM).weight.detach().to(device)

    hdc_model = HDCModel(
        ctx["num_classes"], CNN_CHECKPOINT, device, HD_DIM, model_type=STUDENT_SIZE
    )
    hdc_model.projection.weight.data = rp_weight.clone()
    hdc_model.to(device)

    hdc_trainer = HDCTrainer(
        hdc_model, ctx["num_classes"], device, retrain_epochs=HDC_RETRAIN_EPOCHS
    )
    evaluator = iouEval(ctx["num_classes"], device, ctx["ignore_idx"])
    hdc_trainer.run(ctx["train_loader"], ctx["val_loader"], evaluator)

    hdc_model.sync_class_weights()
    torch.save(
        {
            "projection": hdc_model.projection.state_dict(),
            "classify_weights": hdc_model.classify_weights.data.cpu(),
            "num_classes": ctx["num_classes"],
            "hd_dim": HD_DIM,
            "feat_dim": FEAT_DIM,
            "model_type": STUDENT_SIZE,
            "cnn_checkpoint": CNN_CHECKPOINT,
        },
        HDC_CHECKPOINT,
    )
    print(f"Saved HDC checkpoint to {HDC_CHECKPOINT}")


def compute_proto_distances(protos):
    """Computes pairwise cosine distances between class prototypes."""
    norm_protos = F.normalize(protos, dim=1)
    sim_matrix = norm_protos @ norm_protos.t()
    dist_matrix = 1 - sim_matrix
    return dist_matrix


def main():
    ctx = load_context()
    device = ctx["device"]
    num_classes = ctx["num_classes"]
    train_loader = ctx["train_loader"]
    val_loader = ctx["val_loader"]
    loss_weights = ctx["loss_weights"]
    ignore_idx = ctx["ignore_idx"]
    decay = ctx["arch_cfg"]["train"]["decay"]

    print("\n[INFO] Generating common random HD projection...")
    torch.manual_seed(42)
    common_rp_weight = embeddings.Projection(FEAT_DIM, HD_DIM).weight.detach().to(device)

    random_protos = torch.randn(num_classes, HD_DIM, device=device)
    random_protos = F.normalize(random_protos, dim=1)

    os.makedirs("logs/tezo_student", exist_ok=True)
    os.makedirs("logs/dfa_student", exist_ok=True)
    os.makedirs("logs/baseline", exist_ok=True)

    if TRAIN_TEZO:
        print("\n--- [Branch 1] Training TeZO Teacher (ResNet34) ---")
        tezo_trainer = TeZOTrainer(
            num_classes=num_classes, loss_weights=loss_weights, hd_dim=HD_DIM, feat_dim=FEAT_DIM,
            log_dir="logs/tezo_teacher", device=device,
            lr=decay["lr"], wup_epochs=decay["wup_epochs"], lr_decay=decay["lr_decay"],
            steps_per_epoch=len(train_loader),
        )
        tezo_trainer.rp_weight = common_rp_weight
        tezo_trainer.set_class_protos(random_protos)
        tezo_trainer.train(train_loader, TEACHER_FEATURE_EXTRACTOR_EPOCHS)

        print(f"\n--- [Branch 1] Distilling TeZO Teacher -> Student ({STUDENT_SIZE}) ---")
        distiller_tezo = RKDDistiller(tezo_trainer.model, device)
        tezo_student = distiller_tezo.distill(
            model_size=STUDENT_SIZE, dataloader=train_loader,
            epochs=STUDENT_FEATURE_EXTRACTOR_EPOCHS, num_classes=num_classes,
        )
        torch.save({"state_dict": tezo_student.state_dict()}, "logs/tezo_student/SENet")
    else:
        print("\n--- [Branch 1] SKIPPED: TeZO Training & Distillation ---")

    if TRAIN_DFA:
        print("\n--- [Branch 2] Training DFA Teacher (ResNet34) ---")
        dfa_trainer = DFATrainer(
            num_classes=num_classes, loss_weights=loss_weights, hd_dim=HD_DIM, feat_dim=FEAT_DIM,
            log_dir="logs/dfa_teacher", device=device,
            lr=decay["lr"], wup_epochs=decay["wup_epochs"], lr_decay=decay["lr_decay"],
            steps_per_epoch=len(train_loader),
        )
        dfa_trainer.rp_weight = common_rp_weight
        dfa_trainer.set_class_protos(random_protos)
        dfa_trainer.train(train_loader, TEACHER_FEATURE_EXTRACTOR_EPOCHS)

        print(f"\n--- [Branch 2] Distilling DFA Teacher -> Student ({STUDENT_SIZE}) ---")
        distiller_dfa = RKDDistiller(dfa_trainer.model, device)
        dfa_student = distiller_dfa.distill(
            model_size=STUDENT_SIZE, dataloader=train_loader,
            epochs=STUDENT_FEATURE_EXTRACTOR_EPOCHS, num_classes=num_classes,
        )
        torch.save({"state_dict": dfa_student.state_dict()}, "logs/dfa_student/SENet")
    else:
        print("\n--- [Branch 2] SKIPPED: DFA Training & Distillation ---")

    if TRAIN_BASELINE:
        print(f"\n--- [Branch 3] Training Baseline Student ({STUDENT_SIZE}) ---")
        baseline_model = build_student_model(num_classes)
        baseline_trainer = CNNTrainer(
            num_classes=num_classes, loss_weights=loss_weights, log_dir="logs/baseline",
            device=device, model=baseline_model, aux_loss=False,
        )
        baseline_trainer.train(train_loader, TEACHER_FEATURE_EXTRACTOR_EPOCHS)
        net = baseline_trainer.model.module if hasattr(baseline_trainer.model, "module") else baseline_trainer.model
        torch.save({"state_dict": net.state_dict()}, "logs/baseline/SENet")
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
    results.append("\n" + "=" * 95)
    results.append("FINAL ABLATION STUDY: GENERALIZATION & PROTOTYPE SEPARATION")
    results.append("=" * 95)
    results.append(f"{'Model Strategy':<25} | {'Val mIoU':<10} | {'Avg Proto Dist':<15}")
    results.append("-" * 95)

    dist_tezo = compute_proto_distances(protos_tezo) if protos_tezo is not None else None
    dist_dfa = compute_proto_distances(protos_dfa) if protos_dfa is not None else None
    dist_base = compute_proto_distances(protos_base) if protos_base is not None else None

    if acc_tezo is not None:
        results.append(f"{'TeZO -> Distill':<25} | {acc_tezo:<10.4f} | {dist_tezo.mean():<15.4f}")
    if acc_dfa is not None:
        results.append(f"{'DFA -> Distill':<25} | {acc_dfa:<10.4f} | {dist_dfa.mean():<15.4f}")
    if acc_base is not None:
        results.append(f"{'Baseline (Standard)':<25} | {acc_base:<10.4f} | {dist_base.mean():<15.4f}")

    results.append("-" * 95)
    results.append("\nPairwise Distance Comparison:")
    header = f"{'Classes':<10} | {'TeZO-Dist':<10} | {'DFA-Dist':<10} | {'Baseline':<10} | {'T-B Diff':<10}"
    results.append(header)
    results.append("-" * len(header))

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dt = dist_tezo[i, j].item() if dist_tezo is not None else float("nan")
            dd = dist_dfa[i, j].item() if dist_dfa is not None else float("nan")
            db = dist_base[i, j].item() if protos_base is not None else float("nan")
            diff = (dt - db) if (dist_tezo is not None and dist_base is not None) else float("nan")
            results.append(f"{i:02d} vs {j:02d}:   | {dt:<10.4f} | {dd:<10.4f} | {db:<10.4f} | {diff:<10.4f}")

    final_output = "\n".join(results)
    print(final_output)

    with open(os.path.expanduser("experiment_final_ablation.log"), "w") as f:
        f.write(final_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KD_RL_HDC training pipeline")
    parser.add_argument("--convert", action="store_true", help="Convert raw NuScenes mini to SemanticKITTI format")
    parser.add_argument("--train-cnn", action="store_true", help="Train CNN feature extractor (saves to logs/SENet)")
    parser.add_argument("--train-hdc", action="store_true", help="Train HDC on frozen CNN features (saves to logs/hdc.pth)")
    parser.add_argument("--all", action="store_true", help="Convert, train CNN, then train HDC")
    args = parser.parse_args()

    if args.all:
        convert_dataset()
        train_cnn()
        train_hdc()
    elif args.convert:
        convert_dataset()
    elif args.train_cnn and args.train_hdc:
        train_cnn()
        train_hdc()
    elif args.train_cnn:
        train_cnn()
    elif args.train_hdc:
        train_hdc()
    else:
        main()
