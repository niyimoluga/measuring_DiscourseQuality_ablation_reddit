import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import DebertaV2Tokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

# ==========================================
# PHASE 1: REPRODUCIBILITY & HARDWARE CONFIG
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# Verify device is pinned strictly to empty GPU 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"🔒 Isolated execution device targeted: {device}")

# Use existing W&B login / WANDB_API_KEY environment variable; never hard-code credentials.
if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"])

# ==========================================
# PHASE 2: METRIC CONFIGURATIONS (THE MATRIX)
# ==========================================
# Define the macro-categories exactly as split by your empirical correlation dendrogram
HOSTILITY_COLS = [
    "P1", "P3", "A3−", "P8", "P6", "P5", "P7", "D11", "E7", "A1", "A2", 
    "F7−", "A6", "D3−", "P10", "D9", "F9", "A4", "P4", "A5−", "F13", 
    "D6−", "D4−", "D7−", "E4−", "P11", "P2", "I3−", "I8","D5−"
]

DELIBERATION_COLS = [
    "E2", "E3", "D4+", "I1", "I4", "I7", "I3+", "D1a", "D1b", "A13", 
    "I9", "A5+", "D3+", "A11", "A12", "D5+", "D7+", "A3+", "D6+", 
    "E1", "F7+", "D8","E4+"
]

# Ensure precise matching to fix unicode hyphen strings from the dataset
HOSTILITY_COLS = [c.replace("−", "\u2212") for c in HOSTILITY_COLS]
DELIBERATION_COLS = [c.replace("−", "\u2212") for c in DELIBERATION_COLS]

from dimension_manifest import DIMENSION_SETS

TARGET_COLS = HOSTILITY_COLS + DELIBERATION_COLS
STD_COLS = [f"{col}_std" for col in TARGET_COLS]
IS_HOSTILITY_TASK = {col: (col in HOSTILITY_COLS) for col in TARGET_COLS}


def configure_dimensions(num_dimensions):
    """Set the selected balanced task subset from the shared manifest."""
    global HOSTILITY_COLS, DELIBERATION_COLS, TARGET_COLS, STD_COLS, IS_HOSTILITY_TASK
    if num_dimensions not in DIMENSION_SETS:
        raise ValueError(f"num_dimensions must be one of {sorted(DIMENSION_SETS)}")
    subset = DIMENSION_SETS[num_dimensions]
    HOSTILITY_COLS = list(subset["hostility"])
    DELIBERATION_COLS = list(subset["deliberation"])
    TARGET_COLS = HOSTILITY_COLS + DELIBERATION_COLS
    STD_COLS = [f"{col}_std" for col in TARGET_COLS]
    IS_HOSTILITY_TASK = {col: col in HOSTILITY_COLS for col in TARGET_COLS}
    assert len(TARGET_COLS) == num_dimensions
    print(f"Selected {num_dimensions} dimensions | Hostility={len(HOSTILITY_COLS)} | Deliberation={len(DELIBERATION_COLS)}")
    print(f"Pair ranks: {subset['pair_ranks']}")
    print(f"Targets in model output order: {TARGET_COLS}")

# ==========================================
# PHASE 3: CUSTOM PLE MODEL DEFINITION
# ==========================================
class ExpertBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    def forward(self, x, expert_outputs):
        # expert_outputs: [batch_size, num_experts, hidden_dim]
        gate_scores = torch.softmax(self.gate(x), dim=-1) # [batch_size, num_experts]
        gate_scores = gate_scores.unsqueeze(1) # [batch_size, 1, num_experts]
        output = torch.bmm(gate_scores, expert_outputs) # [batch_size, 1, hidden_dim]
        return output.squeeze(1)

class HierarchicalPLEModel(nn.Module):
    def __init__(self, backbone_name="microsoft/deberta-v3-large"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        emb_dim = self.encoder.config.hidden_size
        expert_dim = 256
        
        # --- LAYER 1 EXPERTS ---
        self.num_hostility_experts = 4
        self.num_deliberation_experts = 4
        self.num_shared_experts_l1 = 4
        
        self.hostility_experts_l1 = nn.ModuleList([ExpertBlock(emb_dim, expert_dim) for _ in range(self.num_hostility_experts)])
        self.deliberation_experts_l1 = nn.ModuleList([ExpertBlock(emb_dim, expert_dim) for _ in range(self.num_deliberation_experts)])
        self.shared_experts_l1 = nn.ModuleList([ExpertBlock(emb_dim, expert_dim) for _ in range(self.num_shared_experts_l1)])
        
        # --- LAYER 1 GATES ---
        self.hostility_gate_l1 = GatingNetwork(emb_dim, self.num_hostility_experts + self.num_shared_experts_l1)
        self.deliberation_gate_l1 = GatingNetwork(emb_dim, self.num_deliberation_experts + self.num_shared_experts_l1)
        self.shared_gate_l1 = GatingNetwork(emb_dim, self.num_hostility_experts + self.num_deliberation_experts + self.num_shared_experts_l1)
        
        # --- LAYER 2 TASK REFINEMENT ---
        self.task_experts = nn.ModuleDict()
        self.task_gates = nn.ModuleDict()
        
        # FIX: Project L1 outputs through unique mappings to align representations
        self.macro_projections = nn.ModuleDict()
        self.shared_projections = nn.ModuleDict()
        
        self.towers = nn.ModuleDict()
        
        for col in TARGET_COLS:
            self.task_experts[col] = ExpertBlock(expert_dim, expert_dim)
            self.macro_projections[col] = nn.Linear(expert_dim, expert_dim)
            self.shared_projections[col] = nn.Linear(expert_dim, expert_dim)
            
            # Each task gate combines: Task Expert + Projected Category Stream + Projected Shared Stream
            self.task_gates[col] = GatingNetwork(expert_dim, 3)
            self.towers[col] = nn.Sequential(
                nn.Linear(expert_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        x = sum_embeddings / sum_mask
        
        # --- LAYER 1 ROUTING ENGINE ---
        h_exp_out = torch.stack([exp(x) for exp in self.hostility_experts_l1], dim=1)
        d_exp_out = torch.stack([exp(x) for exp in self.deliberation_experts_l1], dim=1)
        s_exp_out = torch.stack([exp(x) for exp in self.shared_experts_l1], dim=1)
        
        hostility_macro = self.hostility_gate_l1(x, torch.cat([h_exp_out, s_exp_out], dim=1))
        deliberation_macro = self.deliberation_gate_l1(x, torch.cat([d_exp_out, s_exp_out], dim=1))
        shared_macro = self.shared_gate_l1(x, torch.cat([h_exp_out, d_exp_out, s_exp_out], dim=1))
        
        # --- LAYER 2 PROGRESSIVE SEPARATION LAYER ---
        final_predictions = {}
        for col in TARGET_COLS:
            # 1. Compute specialized task expert representation
            task_exp = self.task_experts[col](shared_macro).unsqueeze(1)
            
            # 2. Select raw macro routing stream lane dynamically
            raw_category = hostility_macro if IS_HOSTILITY_TASK[col] else deliberation_macro
            
            # 3. FIX: Apply the linear transformations to safely bridge abstraction layers
            category_lane = torch.relu(self.macro_projections[col](raw_category)).unsqueeze(1)
            shared_lane = torch.relu(self.shared_projections[col](shared_macro)).unsqueeze(1)
            
            # 4. Concatenate and pass into your Batch Matrix Multiplication routing engine
            combined_experts = torch.cat([task_exp, category_lane, shared_lane], dim=1)
            gated_representation = self.task_gates[col](shared_macro, combined_experts)
            
            final_predictions[col] = self.towers[col](gated_representation)
            
        return torch.cat([final_predictions[col] for col in TARGET_COLS], dim=1)
# ==========================================
# PHASE 4: LOSS CONFIGURATIONS (UNCERTAINTY)
# ==========================================
def ple_loss(logits, labels, stds, median_var: float = 1.0):
    """
    Uncertainty-weighted MSE loss with capped weights (Tim's fix).

    The original 1e-5 epsilon caused weights to explode to ~100,000 when
    std ≈ 0, making a handful of perfect-agreement examples dominate training
    (the 'megaphone effect'). Fix: use the median variance across the training
    set as the epsilon, so weights stay in a gentle 0.5×–2× range rather than
    1×–100,000×.

    median_var is computed once from the training labels before training starts
    and passed in here so the denominator is always sensibly bounded.
    """
    weights = 1.0 / (stds ** 2 + median_var)   # bounded: max weight ≈ 2× normal
    squared_errors = (logits - labels) ** 2
    return (squared_errors * weights).mean(), squared_errors

def plain_mse_loss(logits, labels):
    """Unweighted MSE shared with the plain-head ablation."""
    squared_errors = (logits - labels) ** 2
    return squared_errors.mean(), squared_errors

# ==========================================
# PHASE 5: RUNTIME EXECUTION PIPELINE
# ==========================================
def custom_data_collator(features):
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
        "stds": torch.stack([f["stds"] for f in features]),
    }
def main():
    parser = argparse.ArgumentParser(description="PLE balanced dimension-ablation smoke test")
    parser.add_argument("--num_dimensions", type=int, choices=[2, 4, 8, 16, 32], required=True)
    args = parser.parse_args()
    configure_dimensions(args.num_dimensions)
    run_name = f"ple_{args.num_dimensions}D"

    run = wandb.init(
        entity="niyimoluga85-universty-of-technology-graz",
        project="53 Dimension discourse",
        name=run_name,
        config={
            "learning_rate": 2e-5,
            "architecture": "PLE",
            "experiment_group": "dimension_ablation",
            "task_count": args.num_dimensions,
            "loss": "unweighted_mse",
            "selected_dimensions": TARGET_COLS,
            "smoke_test": True,
            "backbone": "microsoft/deberta-v3-large",
            "dataset": "lanretto/reddit-political-discourse-quality",
            "epochs": 1,
            "batch_size": 8
        }
    )

    print("📖 Loading dataset and initializing tokenizer...")
    raw_dataset = load_dataset("lanretto/reddit-political-discourse-quality")
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")

    def preprocess_function(examples):
        # Build context-aware string formatting logic natively
        inputs = [
            f"[TITLE] {t} [PARENT] {p} [COMMENT] {c}"
            for t, p, c in zip(examples["op_title"], examples["parent_text"], examples["comment_text"])
        ]
        tokenized = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        
        # Construct exact stacked regression and error matrices
        labels_matrix = np.column_stack([examples[col] for col in TARGET_COLS])
        stds_matrix = np.column_stack([examples[s_col] for s_col in STD_COLS])
        
        tokenized["labels"] = labels_matrix.tolist()
        tokenized["stds"] = stds_matrix.tolist()
        return tokenized

   # --- STEP 1: SPLIT THE RAW DATASET FIRST ---
    # Three-way split: 80% train / 10% validation / 10% test
    # Validation is used for early stopping and model selection.
    # Test set is held out and only used for final reported numbers.
    print("✂️  Creating 80/10/10 train/val/test split (seed=42)...")
    full_train = raw_dataset["train"]
    # Step 1: carve off 10% as test
    tmp = full_train.train_test_split(test_size=0.1, seed=42)
    test_raw  = tmp["test"]
    # Step 2: from the remaining 90%, carve off ~11.1% → gives 10% of total as val
    tmp2 = tmp["train"].train_test_split(test_size=0.1111, seed=42)
    train_raw = tmp2["train"]
    eval_raw  = tmp2["test"]
    print(f"   Train: {len(train_raw)} | Val: {len(eval_raw)} | Test: {len(test_raw)} examples")

    # --- STEP 2: TOKENIZE THE SPLITS DIRECTLY ---
    print("⚡ Tokenizing text splits across discourse streams...")
    train_data = train_raw.map(preprocess_function, batched=True, remove_columns=train_raw.column_names)
    eval_data = eval_raw.map(preprocess_function, batched=True, remove_columns=eval_raw.column_names)

    assert "labels" in train_data.column_names, f"'labels' missing! Got: {train_data.column_names}"
    assert "stds"   in train_data.column_names, f"'stds' missing! Got: {train_data.column_names}"

    # Tokenize test set
    test_data = test_raw.map(preprocess_function, batched=True, remove_columns=test_raw.column_names)

    # Drop token_type_ids — not used by the model and causes DataLoader segfaults
    keep_cols = ["input_ids", "attention_mask", "labels", "stds"]
    train_data = train_data.select_columns(keep_cols)
    eval_data  = eval_data.select_columns(keep_cols)
    test_data  = test_data.select_columns(keep_cols)

    # Lock format so Trainer cannot strip our custom 2D label columns
    train_data.set_format(type="torch", columns=keep_cols)
    eval_data.set_format(type="torch", columns=keep_cols)
    test_data.set_format(type="torch", columns=keep_cols)
    print(f"✅ Dataset format locked: {keep_cols}")
    train_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "stds"]
    )

    eval_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "stds"]
    )

    # --- STEP 3: CLEAN COLLATOR ENGINE ---
    def custom_data_collator(features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "stds": torch.stack([f["stds"] for f in features]),
        }
    model = HierarchicalPLEModel().to(device)
    print("Model created successfully")

    # Sanity check
    batch = custom_data_collator([train_data[0], train_data[1]])
    print("\n=== MANUAL BATCH TEST ===")
    for k, v in batch.items():
        print(k, v.shape, v.dtype)
    with torch.no_grad():
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    print("Forward pass output shape:", out.shape)

    # ==========================================
    # PLAIN PYTORCH TRAINING LOOP
    # Replaces HuggingFace Trainer to avoid
    # accelerate/CUDA fork segfaults
    # ==========================================
    BATCH_SIZE     = 8
    LR             = 2e-5
    WEIGHT_DECAY   = 0.01
    LOG_STEPS      = 10
    EVAL_STEPS     = 100
    SAVE_STEPS     = 200
    NUM_EPOCHS     = 4
    SMOKE_TEST     = False  # Set False and restore normal settings for full experiments.
    
    run_name = f"ple_{len(TARGET_COLS)}D"
    CHECKPOINT_DIR = f"/data/nfs/larry/runs_time/{run_name}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=custom_data_collator, num_workers=0, pin_memory=False
    )
    eval_loader = DataLoader(
        eval_data, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=custom_data_collator, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=custom_data_collator, num_workers=0, pin_memory=False
    )

    # The dimension-ablation comparison intentionally uses unweighted MSE.
    # The dataset's *_std columns may remain in the batch for source-code
    # compatibility, but they are not used in the loss.
    print("📊 Loss configured as unweighted MSE; uncertainty weights are disabled.")

    # ── Compute per-dimension training label means (for R²) ───────────────
    # R² = 1 - SS_res/SS_tot where SS_tot uses the TRAINING mean, not eval mean.
    # This matches the majority baseline which predicts the training mean.
    print("📊 Computing per-dimension training label means for R² computation...")
    train_label_means = torch.tensor(
        [np.mean(train_raw[col]) for col in TARGET_COLS],
        dtype=torch.float32
    ).to(device)
    print(f"   Label means range: [{train_label_means.min():.3f}, {train_label_means.max():.3f}]")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    dim_loss_acc   = torch.zeros(len(TARGET_COLS))
    dim_loss_steps = 0
    grad_norm_acc   = 0.0   # running sum of pre-clip gradient norms
    grad_norm_count = 0      # number of steps accumulated
    log_history    = []
    global_step    = 0
    best_eval_loss = float("inf")

    print("\n🚀 Commencing PLE dimension-ablation training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            stds           = batch["stds"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss, sq_err = plain_mse_loss(logits, labels)
            loss.backward()

            # Capture pre-clip gradient norm — this is the raw signal before clamping.
            # High values here explain the loss spikes (exploding gradients on low-std batches).
            pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if dim_loss_acc.device != sq_err.device:
                    dim_loss_acc = dim_loss_acc.to(sq_err.device)
                dim_loss_acc += sq_err.mean(dim=0).detach()
                dim_loss_steps += 1

            # Accumulate grad norm for averaging over LOG_STEPS
            grad_norm_acc   = grad_norm_acc + pre_clip_grad_norm
            grad_norm_count += 1

            global_step += 1
            if global_step % LOG_STEPS == 0:
                avg_dim       = dim_loss_acc / dim_loss_steps
                avg_grad_norm = grad_norm_acc / grad_norm_count
                log_entry = {
                    "step":                      global_step,
                    "epoch":                     epoch + 1,
                    "train/loss":                loss.item(),
                    "train/lr":                  scheduler.get_last_lr()[0],
                    "train/grad_norm":           avg_grad_norm,
                    "train/true_avg_mse_OVERALL": avg_dim.mean().item(),
                }
                for i, col in enumerate(TARGET_COLS):
                    log_entry[f"train/true_avg_mse_{col}"] = avg_dim[i].item()
                wandb.log(log_entry, step=global_step)
                log_history.append(log_entry)
                pd.DataFrame(log_history).to_csv(f"thesis_cache_{run_name}.csv", index=False)
                dim_loss_acc.zero_()
                dim_loss_steps  = 0
                grad_norm_acc   = 0.0
                grad_norm_count = 0
                print(f"  Step {global_step}/{total_steps} | loss: {loss.item():.4f} | grad_norm: {avg_grad_norm:.4f}")

            if global_step % EVAL_STEPS == 0:
                model.eval()
                eval_losses   = []
                eval_dim_acc  = torch.zeros(len(TARGET_COLS))   # SS_residual accumulator
                eval_ss_tot   = torch.zeros(len(TARGET_COLS))   # SS_total accumulator
                eval_steps    = 0
                with torch.no_grad():
                    for eval_batch in eval_loader:
                        ei  = eval_batch["input_ids"].to(device)
                        ea  = eval_batch["attention_mask"].to(device)
                        el  = eval_batch["labels"].to(device)
                        es  = eval_batch["stds"].to(device)
                        elo = model(ei, ea)
                        eloss, esq_err = plain_mse_loss(elo, el)
                        eval_losses.append(eloss.item())
                        if eval_dim_acc.device != esq_err.device:
                            eval_dim_acc = eval_dim_acc.to(esq_err.device)
                            eval_ss_tot  = eval_ss_tot.to(esq_err.device)
                        # SS_residual: squared errors per dimension
                        eval_dim_acc += esq_err.sum(dim=0).detach()
                        # SS_total: (true - training_mean)² per dimension
                        eval_ss_tot  += ((el - train_label_means) ** 2).sum(dim=0).detach()
                        eval_steps   += el.shape[0]   # count examples, not batches

                mean_eval    = sum(eval_losses) / len(eval_losses)
                avg_mse_dim  = eval_dim_acc / eval_steps         # per-dim MSE
                # R² per dimension: 1 - SS_res/SS_tot (clamped so negatives show as-is)
                r2_per_dim   = 1.0 - (eval_dim_acc / (eval_ss_tot + 1e-10))

                eval_log = {
                    "step":                      global_step,
                    "epoch":                     epoch + 1,
                    "eval/loss":                 mean_eval,
                    "eval/true_avg_mse_OVERALL": avg_mse_dim.mean().item(),
                    "eval/r2_OVERALL":           r2_per_dim.mean().item(),
                }
                for i, col in enumerate(TARGET_COLS):
                    eval_log[f"eval/true_avg_mse_{col}"] = avg_mse_dim[i].item()
                    eval_log[f"eval/r2_{col}"]           = r2_per_dim[i].item()

                wandb.log(eval_log, step=global_step)
                log_history.append(eval_log)
                pd.DataFrame(log_history).to_csv(f"thesis_cache_{run_name}.csv", index=False)

                r2_overall = r2_per_dim.mean().item()
                print(f"  ✅ Eval @ step {global_step}: loss={mean_eval:.4f} | "
                      f"MSE={avg_mse_dim.mean().item():.4f} | R²={r2_overall:.4f}")
                if mean_eval < best_eval_loss:
                                    best_eval_loss = mean_eval
                                    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                                    print(f"  💾 New best model saved (loss={mean_eval:.4f} | R²={r2_overall:.4f})")
                model.train()

            if global_step % SAVE_STEPS == 0:
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_step{global_step}.pt"))

            # Smoke-test stop is deliberately AFTER logging, eval and checkpointing.
            if SMOKE_TEST and global_step >= 1:
                print("🧪 Smoke test complete: stopping after one fully logged/evaluated/saved step.")
                break

    print("\n✅ Run Completed Successfully!")

    # ==========================================
    # PHASE 6: FINAL HELD-OUT TEST EVALUATION
    # ==========================================
    # The test split has been untouched until now — these are the numbers to report.
    print("\n🧪 Evaluating on held-out test set...")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt))
        print(f"   Loaded best checkpoint: {best_ckpt}")
    else:
        print("   ⚠️  No best_model.pt found — using final in-memory weights.")

    model.eval()
    test_losses  = []
    test_dim_acc = torch.zeros(len(TARGET_COLS), device=device)   # SS_residual
    test_ss_tot  = torch.zeros(len(TARGET_COLS), device=device)   # SS_total
    test_n       = 0
    with torch.no_grad():
        for test_batch in test_loader:
            ti  = test_batch["input_ids"].to(device)
            ta  = test_batch["attention_mask"].to(device)
            tl  = test_batch["labels"].to(device)
            ts  = test_batch["stds"].to(device)
            tlo = model(ti, ta)
            tloss, tsq_err = plain_mse_loss(tlo, tl)
            test_losses.append(tloss.item())
            test_dim_acc += tsq_err.sum(dim=0).detach()
            test_ss_tot  += ((tl - train_label_means) ** 2).sum(dim=0).detach()
            test_n       += tl.shape[0]

    test_mse_dim = test_dim_acc / test_n
    test_r2_dim  = 1.0 - (test_dim_acc / (test_ss_tot + 1e-10))

    test_log = {
        "step":                     global_step,
        "epoch":                    NUM_EPOCHS,
        "test/loss":                sum(test_losses) / len(test_losses),
        "test/true_avg_mse_OVERALL": test_mse_dim.mean().item(),
        "test/r2_OVERALL":          test_r2_dim.mean().item(),
    }
    for i, col in enumerate(TARGET_COLS):
        test_log[f"test/true_avg_mse_{col}"] = test_mse_dim[i].item()
        test_log[f"test/r2_{col}"]           = test_r2_dim[i].item()

    wandb.log(test_log, step=global_step)
    log_history.append(test_log)
    pd.DataFrame(log_history).to_csv(f"thesis_cache_{run_name}.csv", index=False)
    print(f"🧪 TEST ({test_n} examples) | loss={test_log['test/loss']:.4f} | "
          f"MSE={test_log['test/true_avg_mse_OVERALL']:.4f} | "
          f"R²={test_log['test/r2_OVERALL']:.4f}")

    # ==========================================
    # PHASE 7: POST-TRAINING VISUALIZATIONS
    # ==========================================
    print("\n📊 Packaging category metric grids for upload...")
    history = log_history  # populated by the plain PyTorch training loop above

    # Generate charts organized nicely by their structural groups
    plot_groups = {
        "Hostility and Identity Metrics": HOSTILITY_COLS,
        "Deliberation and Logic Metrics": DELIBERATION_COLS
    }

    for title, cols in plot_groups.items():
        num_plots = len(cols)
        ncols = 4
        nrows = int(np.ceil(num_plots / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows * 4))
        axes = axes.flatten()

        for idx, col_name in enumerate(cols):
            ax = axes[idx]
            steps = [log['step'] for log in history if f'train/true_avg_mse_{col_name}' in log]
            losses = [log[f'train/true_avg_mse_{col_name}'] for log in history if f'train/true_avg_mse_{col_name}' in log]

            if steps:
                ax.plot(steps, losses, label='Train (True Avg)', color='#2c3e50', linewidth=2)
                ax.set_title(f"{col_name}", fontsize=12, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlabel('Steps')
                ax.set_ylabel('MSE')

        # Hide extra grid slots cleanly
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        wandb.log({f"Evaluation/Grid_{title.replace(' ', '_')}": wandb.Image(fig)})
        plt.close(fig)

    run.finish()
    print("🏁 Training artifacts and plot collections are completely uploaded to your workspace!")

if __name__ == "__main__":
    main()