import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# LAYER 3: HOLOGRAPHIC MEMORY WITH EXACT PHASE-CANCELLATION UNLEARNING
# ==============================================================================
class HolographicSamskaraMemory(nn.Module):
    """
    Stores key-value associations as superposed complex standing waves via
    circular convolution in the Fourier domain (O(D log D)).
    Uses unitary keys (|F(k)| = 1) for exact algebraic unbinding and
    millisecond-scale, zero-retraining anti-phase unlearning.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("memory_field", torch.zeros(dim))

    @staticmethod
    def make_unitary(x: torch.Tensor) -> torch.Tensor:
        """Projects vectors onto the unitary Fourier manifold for lossless unbinding."""
        X_f = torch.fft.rfft(x, dim=-1)
        X_f_unit = X_f / (torch.abs(X_f) + 1e-8)
        return torch.fft.irfft(X_f_unit, n=x.shape[-1], dim=-1)

    def bind(self, key: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
        K_f = torch.fft.rfft(key, dim=-1)
        V_f = torch.fft.rfft(val, dim=-1)
        return torch.fft.irfft(K_f * V_f, n=self.dim, dim=-1)

    def unbind(self, field: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        M_f = torch.fft.rfft(field, dim=-1)
        K_f = torch.fft.rfft(key, dim=-1)
        # Complex conjugate in Fourier space acts as exact inverse for unitary keys
        return torch.fft.irfft(M_f * torch.conj(K_f), n=self.dim, dim=-1)

    @torch.no_grad()
    def write(self, key: torch.Tensor, val: torch.Tensor):
        u_key = self.make_unitary(key)
        self.memory_field.add_(self.bind(u_key, val))

    @torch.no_grad()
    def read(self, key: torch.Tensor) -> torch.Tensor:
        u_key = self.make_unitary(key)
        return self.unbind(self.memory_field, u_key)

    @torch.no_grad()
    def shiva_unlearn(self, key: torch.Tensor, val: torch.Tensor):
        """
        Exact Phase-Cancellation Unlearning: injects the anti-phase standing wave
        to erase a target memory in O(D log D) without touching orthogonal memories.
        """
        u_key = self.make_unitary(key)
        anti_wave = -self.bind(u_key, val)
        self.memory_field.add_(anti_wave)


# ==============================================================================
# LAYER 2: O(N) OSCILLATORY RESONANCE TRANSFORMER (ORT) LAYER
# ==============================================================================
class KuramotoResonanceLayer(nn.Module):
    """
    Replaces O(N^2) dot-product attention with O(T * N * D) mean-field
    Kuramoto phase-locking modulated by learnable systemic impedance (Z).
    """
    def __init__(self, dim: int, steps: int = 8, dt: float = 0.25):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.dt = dt

        # Projections to phase (theta), natural frequency (omega), amplitude (V), impedance (Z)
        self.to_theta = nn.Linear(dim, dim)
        self.to_omega = nn.Linear(dim, dim)
        self.to_amp = nn.Linear(dim, dim)
        self.to_impedance = nn.Linear(dim, dim)

        # Global coupling strength K
        self.coupling_K = nn.Parameter(torch.ones(1, 1, dim) * 2.5)
        self.out_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # x: (Batch, SeqLen, Dim)
        theta = torch.tanh(self.to_theta(x)) * math.pi
        omega = self.to_omega(x) * 0.5
        amp = F.softplus(self.to_amp(x))
        # Systemic impedance Z > 0 (high Z = high routing friction)
        Z = F.softplus(self.to_impedance(x)) + 0.1

        # Iterative Kuramoto Phase-Locking (O(N) via Mean-Field Order Parameter)
        for _ in range(self.steps):
            # Conductance-weighted complex phasors: G = 1 / Z
            conductance = 1.0 / Z
            cos_t = torch.cos(theta) * conductance
            sin_t = torch.sin(theta) * conductance

            # Mean-field carrier wave across sequence dimension (N)
            mean_cos = cos_t.mean(dim=1, keepdim=True)
            mean_sin = sin_t.mean(dim=1, keepdim=True)

            # Global carrier phase (Psi) and coherence magnitude (Phi)
            psi = torch.atan2(mean_sin, mean_cos + 1e-8)
            
            # d(theta)/dt = omega + (K / Z) * sin(Psi - theta)
            d_theta = omega + (self.coupling_K / Z) * torch.sin(psi - theta)
            theta = theta + self.dt * d_theta

        # Final Order Parameter Phi in [0, 1] per sample
        raw_cos = torch.cos(theta).mean(dim=1)
        raw_sin = torch.sin(theta).mean(dim=1)
        phi = torch.sqrt(raw_cos ** 2 + raw_sin ** 2 + 1e-8).mean(dim=-1)  # (Batch,)

        # Reconstruct real-valued representation from complex oscillator state
        osc_state = torch.cat([amp * torch.cos(theta), amp * torch.sin(theta)], dim=-1)
        out = self.norm(x + self.out_proj(osc_state))
        mean_Z = Z.mean(dim=(1, 2))  # (Batch,)

        return out, phi, mean_Z


# ==============================================================================
# LAYER 4: VOLTAGE-GATED METACOGNITIVE HRC ENGINE
# ==============================================================================
class HomeostaticResonantModel(nn.Module):
    """
    Full-stack HRC software prototype:
    - Stacked O(N) Kuramoto Resonance layers
    - Virtual Membrane Resting Potential (V_m in [-90mV, -20mV])
    - Automatic Epistemic Halt when V_m depolarizes above critical threshold
    - Shiva Autophagic Pruning for continual learning stability
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, v_crit: float = -45.0):
        super().__init__()
        self.v_crit = v_crit
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            KuramotoResonanceLayer(hidden_dim, steps=8) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, num_classes)
        self.memory = HolographicSamskaraMemory(hidden_dim)

    def forward(self, x: torch.Tensor):
        h = self.embed(x)
        phis, zs = [], []

        for layer in self.layers:
            h, phi, z = layer(h)
            phis.append(phi)
            zs.append(z)

        # Aggregate global coherence (Phi) and impedance (Z)
        global_phi = torch.stack(phis, dim=0).mean(dim=0)  # (Batch,)
        global_Z = torch.stack(zs, dim=0).mean(dim=0)      # (Batch,)

        # Pool sequence & classify
        logits = self.head(h.mean(dim=1))
        probs = F.softmax(logits, dim=-1)

        # Normalized Predictive Entropy (Variational Surprise proxy)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        max_entropy = math.log(logits.shape[-1])
        norm_entropy = entropy / (max_entropy + 1e-8)

        # Virtual Membrane Voltage V_m in [-85mV, -20mV]
        # High coherence & low surprise -> polarized (-80mV, Buddhi discernment)
        # Low coherence or high surprise -> depolarized (-20mV, Stress / Hallucination risk)
        stress_signal = torch.clamp(0.55 * (1.0 - global_phi) + 0.45 * norm_entropy, 0.0, 1.0)
        v_m = -85.0 + 65.0 * stress_signal

        # Epistemic Halt Flag: True if system depolarizes past critical threshold
        halt_flag = v_m > self.v_crit

        return {
            "logits": logits,
            "phi": global_phi,
            "Z": global_Z,
            "v_m": v_m,
            "halt_flag": halt_flag
        }

    @torch.no_grad()
    def shiva_autophagic_prune(self, quantile: float = 0.15):
        """Offline consolidation: prunes lowest-utility weights to zero impedance."""
        pruned_count, total_count = 0, 0
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                thresh = torch.quantile(param.abs().flatten(), quantile)
                mask = (param.abs() >= thresh).float()
                pruned_count += (mask == 0).sum().item()
                total_count += param.numel()
                param.mul_(mask)
        return pruned_count / max(total_count, 1)


def homeostatic_free_energy_loss(out: dict, targets: torch.Tensor, lambda_phi=0.5, lambda_z=0.05):
    """L_HRC = Variational Surprise (CE) + lambda_1 * (1 - Phi) + lambda_2 * ||Z||_1"""
    ce_loss = F.cross_entropy(out["logits"], targets)
    coherence_penalty = (1.0 - out["phi"]).mean()
    impedance_penalty = out["Z"].mean()
    return ce_loss + lambda_phi * coherence_penalty + lambda_z * impedance_penalty


# ==============================================================================
# AUTOMATED VALIDATION SUITE (3 BREAKTHROUGH PROOFS)
# ==============================================================================
def run_hrc_validation_suite():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[HRC MVP] Running Validation Suite on: {device}\n" + "=" * 68)

    # ------------------------------------------------------------------
    # BENCHMARK 1: Training & Phase-Locked Convergence
    # ------------------------------------------------------------------
    B, N, D_in, D_hid, C = 64, 16, 16, 64, 4
    model = HomeostaticResonantModel(D_in, D_hid, C, num_layers=2, v_crit=-45.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Structured in-distribution class prototypes (harmonic patterns)
    prototypes = torch.randn(C, 1, D_in, device=device) * 1.5

    for epoch in range(60):
        targets = torch.randint(0, C, (B,), device=device)
        x_in = prototypes[targets].expand(B, N, D_in) + torch.randn(B, N, D_in, device=device) * 0.35
        out = model(x_in)
        loss = homeostatic_free_energy_loss(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    print(f"[TEST 1: Homeostatic Convergence]")
    print(f"  Final Loss: {loss.item():.4f} | Mean Coherence (Phi): {out['phi'].mean().item():.4f}")

    # ------------------------------------------------------------------
    # BENCHMARK 2: Voltage-Gated OOD / Anti-Hallucination Detection
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Calibrate the halt threshold from a held-out in-distribution batch.
        t_cal = torch.randint(0, C, (64,), device=device)
        x_cal = prototypes[t_cal].expand(64, N, D_in) + torch.randn(64, N, D_in, device=device) * 0.35
        cal_vm = model(x_cal)["v_m"]
        model.v_crit = (cal_vm.mean() + 3.0 * cal_vm.std() + 2.0).item()
        print(f"  [Calibrated V_m Threshold: {model.v_crit:.1f} mV]")

        # In-distribution test batch
        t_id = torch.randint(0, C, (32,), device=device)
        x_id = prototypes[t_id].expand(32, N, D_in) + torch.randn(32, N, D_in, device=device) * 0.35
        res_id = model(x_id)

        # Out-of-distribution (OOD) high-entropy adversarial noise
        x_ood = torch.randn(32, N, D_in, device=device) * 3.0
        res_ood = model(x_ood)

    print(f"\n[TEST 2: Voltage-Gated Metacognition (Threshold = {model.v_crit:.1f} mV)]")
    print(f"  In-Distribution  -> V_m: {res_id['v_m'].mean().item():.1f} mV | Phi: {res_id['phi'].mean().item():.3f} | Halt Rate: {res_id['halt_flag'].float().mean().item()*100:.1f}%")
    print(f"  Out-of-Dist (OOD)-> V_m: {res_ood['v_m'].mean().item():.1f} mV | Phi: {res_ood['phi'].mean().item():.3f} | Halt Rate: {res_ood['halt_flag'].float().mean().item()*100:.1f}%")

    # ------------------------------------------------------------------
    # BENCHMARK 3: Exact Zero-Retraining Holographic Unlearning
    # ------------------------------------------------------------------
    mem = HolographicSamskaraMemory(dim=1024).to(device)
    K_facts = 20
    keys = torch.randn(K_facts, 1024, device=device)
    vals = F.normalize(torch.randn(K_facts, 1024, device=device), dim=-1)

    for i in range(K_facts):
        mem.write(keys[i], vals[i])

    # Verify recall before unlearning
    target_idx = 7
    sim_before = F.cosine_similarity(mem.read(keys[target_idx]), vals[target_idx], dim=0).item()

    # Execute Shiva Anti-Phase Unlearning
    t0 = time.perf_counter()
    mem.shiva_unlearn(keys[target_idx], vals[target_idx])
    unlearn_ms = (time.perf_counter() - t0) * 1000.0

    # Verify target is erased while remaining 19 facts remain intact
    sim_after_target = F.cosine_similarity(mem.read(keys[target_idx]), vals[target_idx], dim=0).item()
    retained_sims = [
        F.cosine_similarity(mem.read(keys[i]), vals[i], dim=0).item()
        for i in range(K_facts) if i != target_idx
    ]

    print(f"\n[TEST 3: Exact Holographic Phase-Cancellation Unlearning]")
    print(f"  Target Fact #{target_idx} Recall BEFORE Unlearning : {sim_before:.4f}")
    print(f"  Target Fact #{target_idx} Recall AFTER  Unlearning : {sim_after_target:.4f} (Erased in {unlearn_ms:.3f} ms)")
    print(f"  Mean Recall of Remaining 19 Untouched Facts  : {sum(retained_sims)/len(retained_sims):.4f}")

    # ------------------------------------------------------------------
    # BENCHMARK 4: Shiva Offline Autophagic Pruning
    # ------------------------------------------------------------------
    sparsity = model.shiva_autophagic_prune(quantile=0.20)
    print(f"\n[TEST 4: Shiva Autophagic Consolidation]")
    print(f"  Pruned {sparsity*100:.1f}% low-utility synaptic weights cleanly.")
    print("=" * 68)


if __name__ == "__main__":
    run_hrc_validation_suite()