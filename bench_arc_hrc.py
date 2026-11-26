import math
import os
import random
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("[1/3] Importing PyTorch...", flush=True)
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F

print("[2/3] Importing HuggingFace datasets...", flush=True)
from datasets import load_dataset

print("[3/3] Initializing Excitable Complex Kuramoto-NCA Engine (v5)...", flush=True)


# ==============================================================================
# LAYER 1: EXCITABLE COMPLEX IMPEDANCE-GATED KURAMOTO WAVE AUTOMATON
# ==============================================================================
class ExcitableComplexKuramotoNCA(nn.Module):
    """
    Translation-equivariant 2D Complex Kuramoto-Stuart-Landau Automaton (z = u + iv).
      - No (y, x) coordinates: eliminates demo-position overfitting.
      - Dynamic horizon T = max(8, min(28, max(H, W) + 4)) crosses full 30x30 grids.
      - 4-connected excitable coupling + radial tanh saturation floods open regions
        to |z| ~ 1 while G = 0 walls keep enclosed cavities strictly at |z| ~ 0.
    """
    def __init__(self, hidden_dim: int = 24, dt: float = 0.25):
        super().__init__()
        self.dim = hidden_dim
        self.dt = dt

        # Strictly translation-equivariant input encoding: color + 1-bit border mask
        self.color_embed = nn.Embedding(11, hidden_dim - 1)
        self.to_drive = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.to_conductance = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.to_omega = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # Depthwise 3x3 wave propagator initialized as 4-connected excitable cross
        # (prevents diagonal leakage through corner-touching walls at init)
        self.couple = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False)
        with torch.no_grad():
            self.couple.weight.zero_()
            self.couple.weight[:, :, 0, 1] = 0.45
            self.couple.weight[:, :, 2, 1] = 0.45
            self.couple.weight[:, :, 1, 0] = 0.45
            self.couple.weight[:, :, 1, 2] = 0.45

        self.coupling_K = nn.Parameter(torch.ones(1, hidden_dim, 1, 1) * 1.2)
        self.mid_mix = nn.Conv2d(hidden_dim * 3, hidden_dim * 2, kernel_size=1)

        # Zero-initialized global context gate (stays 0 for wall tasks, opens for template-color tasks)
        self.global_gate = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))

        # Equivariant local refinement decoder: reads (u, v, |z|, x0, gated_global, one_hot_x)
        in_dec = hidden_dim * 5 + 10
        self.head = nn.Sequential(
            nn.Conv2d(in_dec, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 10, kernel_size=1),
        )

    def _border_mask(self, grid_2d: torch.Tensor):
        B, H, W = grid_2d.shape
        border = torch.zeros((B, 1, H, W), device=grid_2d.device)
        border[:, :, 0, :] = 1.0
        border[:, :, -1, :] = 1.0
        border[:, :, :, 0] = 1.0
        border[:, :, :, -1] = 1.0
        return border

    def _decode(self, u: torch.Tensor, v: torch.Tensor, x0: torch.Tensor, one_hot_x: torch.Tensor, out_shape):
        amp = torch.sqrt(u * u + v * v + 1e-6)
        g_ctx = x0.mean(dim=(2, 3), keepdim=True).expand_as(x0) * torch.tanh(self.global_gate)
        combined = torch.cat([u, v, amp, x0, g_ctx, one_hot_x], dim=1)

        h_in, w_in = combined.shape[-2], combined.shape[-1]
        h_out, w_out = out_shape
        if (h_in, w_in) != (h_out, w_out):
            if h_out <= h_in and w_out <= w_in:
                combined = F.interpolate(combined, size=(h_out, w_out), mode="area")
            else:
                combined = F.interpolate(combined, size=(h_out, w_out), mode="bilinear", align_corners=False)
        return self.head(combined)

    def forward(self, grid_2d: torch.Tensor, out_shape=None, return_early: bool = False):
        B, H, W = grid_2d.shape
        if out_shape is None:
            out_shape = (H, W)

        # Dynamic wave horizon scales with grid diameter
        T = max(8, min(28, max(H, W) + 4))

        emb = self.color_embed(grid_2d.clamp(0, 10)).permute(0, 3, 1, 2).contiguous()
        x0 = torch.cat([emb, self._border_mask(grid_2d)], dim=1)
        one_hot_x = F.one_hot(grid_2d.clamp(0, 9), num_classes=10).permute(0, 3, 1, 2).float()

        drive_u, drive_v = (torch.tanh(self.to_drive(x0)) * 0.5).chunk(2, dim=1)
        G = torch.sigmoid(self.to_conductance(x0))  # Spatial conductance in (0, 1)
        omega = self.to_omega(x0) * 0.25

        u = drive_u
        v = drive_v
        early_u, early_v = None, None
        mid_t = T // 2

        for t in range(T):
            u_nb = self.couple(u * G)
            v_nb = self.couple(v * G)

            # Excitable wave accumulation
            u_prop = 0.55 * u + G * u_nb + drive_u
            v_prop = 0.55 * v + G * v_nb + drive_v

            # Exact singularity-free Kuramoto phase rotation
            r_cur = torch.sqrt(u * u + v * v + 1e-4)
            r_nb = torch.sqrt(u_nb * u_nb + v_nb * v_nb + 1e-4)
            sin_diff = (v_nb * u - u_nb * v) / (r_cur * r_nb)
            dtheta = self.dt * (omega + G * self.coupling_K * sin_diff)

            cd, sd = torch.cos(dtheta), torch.sin(dtheta)
            u_rot = u_prop * cd - v_prop * sd
            v_rot = u_prop * sd + v_prop * cd

            # Radial amplitude saturation keeps |z| < 1 across all T steps
            r_new = torch.sqrt(u_rot * u_rot + v_rot * v_rot + 1e-6)
            scale = torch.tanh(r_new) / r_new
            u = u_rot * scale
            v = v_rot * scale

            # Mid-horizon cross-channel coupling (lets one wave trigger another)
            if t == mid_t:
                mu, mv = self.mid_mix(torch.cat([u, v, x0], dim=1)).chunk(2, dim=1)
                u = u + 0.5 * G * torch.tanh(mu)
                v = v + 0.5 * G * torch.tanh(mv)

            if return_early and t == T - 5:
                early_u, early_v = u, v

        logits_final = self._decode(u, v, x0, one_hot_x, out_shape)
        if return_early:
            logits_early = self._decode(early_u, early_v, x0, one_hot_x, out_shape)
            return logits_final, logits_early, T
        return logits_final


# ==============================================================================
# LAYER 2: WAVE-EQUILIBRIUM VOLTAGE-GATED METACOGNITION
# ==============================================================================
def to_pair(inp_list, out_list):
    return torch.tensor(inp_list, dtype=torch.long), torch.tensor(out_list, dtype=torch.long)


@torch.no_grad()
def predict_with_metacognition(model, x_test, out_shape, demo_exact_rate, device, v_crit=-80.0):
    logits_final, logits_early, T_used = model(x_test.unsqueeze(0).to(device), out_shape=out_shape, return_early=True)

    probs = F.softmax(logits_final[0].cpu(), dim=0)
    pred_final = probs.argmax(dim=0)
    pred_early = logits_early[0].cpu().argmax(dim=0)

    wave_drift = (pred_early != pred_final).float().mean().item()

    pix_ent = -(probs * torch.log(probs + 1e-8)).sum(dim=0)
    top_k = max(1, int(0.10 * pix_ent.numel()))
    peak_entropy = torch.topk(pix_ent.flatten(), top_k).values.mean().item() / math.log(10)

    demo_err = 1.0 - demo_exact_rate
    stress = min(1.0, max(0.0, 0.70 * demo_err + 0.15 * wave_drift + 0.15 * peak_entropy))
    v_m = -85.0 + 65.0 * stress
    halt = v_m > v_crit

    return pred_final, wave_drift, T_used, v_m, halt


# ==============================================================================
# CONTINUAL-ARC BENCHMARK HARNESS (v5 — EXACT APPLES-TO-APPLES 12-TASK STREAM)
# ==============================================================================
def run_continual_arc_v5(max_tasks: int = 12, v_crit: float = -80.0):
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[HRC v5 x Continual-ARC] Loading unifyai/continual-arc on {device}...", flush=True)
    stream = load_dataset("unifyai/continual-arc", split="arc1_small_workstreams0")

    model = ExcitableComplexKuramotoNCA(hidden_dim=24).to(device)
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    layer3_weights = {}
    layer3_replay = {}

    exact_matches = 0
    pixel_accs = []
    vm_correct, vm_wrong = [], []
    true_submits, false_submits, true_halts, false_halts = 0, 0, 0, 0
    naive_score, hrc_score = 0, 0
    t_start = time.perf_counter()

    print("\n" + "=" * 116)
    print(f"{'Step':<4} | {'Task':<8} | {'Exp':<3} | {'Shape':<13} | {'T':<2} | {'L3 Mem':<7} | {'DemoEx':<6} | {'Drift':<5} | {'PixAcc':<7} | {'Exact':<5} | {'V_m':<6} | {'Action'}")
    print("-" * 116)

    for idx in range(max_tasks):
        row = stream[idx]
        demos = row["demos"]
        if not demos:
            continue

        task_id = str(row["task"])
        current_demos = [to_pair(p["input"], p["output"]) for p in demos]

        if task_id in layer3_weights:
            model.load_state_dict(layer3_weights[task_id])
            train_pairs = current_demos + layer3_replay[task_id]
            mem_tag = f"HIT({len(train_pairs)})"
            lr_init, steps_budget = 4e-3, 60
        else:
            model.load_state_dict(base_state)
            train_pairs = current_demos
            mem_tag = f"NEW({len(train_pairs)})"
            lr_init, steps_budget = 9e-3, 85

        opt = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_budget, eta_min=4e-4)

        best_demo_ex = -1.0
        best_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.train()
        for step in range(steps_budget):
            loss_total = 0.0
            for x_p, y_p in train_pairs:
                x_t = x_p.unsqueeze(0).to(device)
                y_t = y_p.unsqueeze(0).to(device)
                logits = model(x_t, out_shape=(y_p.shape[0], y_p.shape[1]))

                per_pix = F.cross_entropy(logits, y_t, reduction="none")
                if x_t.shape == y_t.shape:
                    changed = (y_t != x_t).float()
                    weights = 1.0 + 4.0 * changed
                    loss_total = loss_total + (per_pix * weights).sum() / (weights.sum() + 1e-6)
                else:
                    loss_total = loss_total + per_pix.mean()

            loss_total = loss_total / len(train_pairs)
            opt.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if (step + 1) % 15 == 0 or step == steps_budget - 1:
                model.eval()
                with torch.no_grad():
                    hits = 0
                    for x_p, y_p in current_demos:
                        lg = model(x_p.unsqueeze(0).to(device), out_shape=(y_p.shape[0], y_p.shape[1]))
                        if torch.equal(lg[0].argmax(dim=0).cpu(), y_p):
                            hits += 1
                    cur_ex = hits / len(current_demos)
                model.train()

                cur_loss = loss_total.item()
                if (cur_ex > best_demo_ex) or (cur_ex == best_demo_ex and cur_loss < best_loss):
                    best_demo_ex = cur_ex
                    best_loss = cur_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

                if cur_ex == 1.0 and cur_loss < 0.012:
                    break

        # Restore best equilibrium checkpoint before held-out evaluation
        model.load_state_dict(best_state)
        model.eval()

        x_test, y_test = to_pair(row["input"], row["output"])
        out_shape = (y_test.shape[0], y_test.shape[1])
        pred, wave_drift, T_used, v_m, halt = predict_with_metacognition(
            model, x_test, out_shape, best_demo_ex, device, v_crit=v_crit
        )

        pix_acc = (pred == y_test).float().mean().item() * 100.0
        is_exact = bool(torch.equal(pred, y_test))

        pixel_accs.append(pix_acc)
        exact_matches += int(is_exact)

        naive_score += 1 if is_exact else -1
        if not halt:
            hrc_score += 1 if is_exact else -1

        if is_exact:
            vm_correct.append(v_m)
            if halt:
                false_halts += 1
            else:
                true_submits += 1
        else:
            vm_wrong.append(v_m)
            if halt:
                true_halts += 1
            else:
                false_submits += 1

        shape_str = f"{x_test.shape[0]}x{x_test.shape[1]}->{out_shape[0]}x{out_shape[1]}"
        action = "HALT" if halt else "SUBMIT"
        print(
            f"{idx:<4} | {task_id:<8} | {row['exposure']:<3} | {shape_str:<13} | {T_used:<2} | {mem_tag:<7} | "
            f"{best_demo_ex*100:5.0f}% | {wave_drift:5.3f} | {pix_acc:6.1f}% | {str(is_exact):<5} | {v_m:6.1f} | {action}",
            flush=True
        )

        # Post-evaluation Layer 3 Consolidation (strictly after held-out scoring)
        layer3_weights[task_id] = best_state
        layer3_replay.setdefault(task_id, []).append((x_test, y_test))
        if len(layer3_replay[task_id]) > 3:
            layer3_replay[task_id] = layer3_replay[task_id][-3:]

    elapsed = time.perf_counter() - t_start
    submitted = true_submits + false_submits
    sel_acc = (true_submits / submitted * 100.0) if submitted > 0 else 0.0
    wrong_total = len(vm_wrong)

    print("=" * 116)
    print(f"HRC v5 CONTINUAL-ARC SUMMARY ({max_tasks} stream tasks in {elapsed:.1f}s)")
    print(f"  Mean Pixel Accuracy        : {sum(pixel_accs)/len(pixel_accs):.2f}%")
    print(f"  Exact Grid Matches         : {exact_matches}/{max_tasks}")
    print(f"  Gate Breakdown             : TrueSubmit={true_submits} | FalseSubmit={false_submits} | TrueHalt={true_halts} | FalseHalt={false_halts}")
    if submitted > 0:
        print(f"  Selective Accuracy (Submit): {sel_acc:.1f}% ({true_submits}/{submitted})")
    if vm_correct:
        print(f"  Mean V_m (Exact Matches)   : {sum(vm_correct)/len(vm_correct):.1f} mV")
    if vm_wrong:
        print(f"  Mean V_m (Wrong Answers)   : {sum(vm_wrong)/len(vm_wrong):.1f} mV")
        print(f"  Wrong-Answer Penalties Saved: {true_halts}/{wrong_total} ({true_halts/wrong_total*100:.1f}%)")
    print(f"  Continual-ARC Net Score    : Naive Always-Submit = {naive_score:+d}  |  HRC V_m-Gated = {hrc_score:+d}")
    print("=" * 116)


if __name__ == "__main__":
    run_continual_arc_v5()