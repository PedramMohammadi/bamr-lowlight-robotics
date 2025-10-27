# bamr/train_stage_b.py
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from .models import load_bamr_tiny
from .losses import EdgeLoss, TVLoss, ResponseConsistencyLoss, LoFTRMatchingImprovement
from .data import YOLOImagesDataset, MIDDualViewDataset, make_loader
from .utils import seed_everything, amp_autocast, save_checkpoint

def main():
    parser = argparse.ArgumentParser("Stage-B Task-aware Tuning (LoFTR + Priors)")
    parser.add_argument("--stageA_ckpt", type=str, required=True)
    parser.add_argument("--det_yaml",    type=str, required=True, help="ExDark YAML for detection images (train)")
    parser.add_argument("--mid_root",    type=str, required=True, help="Prepared MID images root (images/{Indoor,Outdoor}/pairXX/viewA|viewB)")
    parser.add_argument("--steps",       type=int, default=7500)
    parser.add_argument("--batch",       type=int, default=4)
    parser.add_argument("--patch",       type=int, default=256, help="uniform crop for compute control")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--ckpt_out",    type=str, default="checkpoints/bamr_stageB/bamr_taskaware.pt")
    # Loss weights (recommended defaults from your best run)
    parser.add_argument("--w_loftr",     type=float, default=1.0)
    parser.add_argument("--w_edge",      type=float, default=0.6)
    parser.add_argument("--w_rc",        type=float, default=0.2)
    parser.add_argument("--tv",          type=float, default=0.0)
    parser.add_argument("--lr",          type=float, default=1e-4)
    args = parser.parse_args()

    seed_everything(123)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Model
    model = load_bamr_tiny(args.stageA_ckpt, device=device, half=(device=="cuda"))
    model.train()

    # Losses
    edge = EdgeLoss(mode="l1")
    rc   = ResponseConsistencyLoss(mean_w=1.0, var_w=1.0)
    tv   = TVLoss(weight=args.tv) if args.tv > 0 else None
    loftr = LoFTRMatchingImprovement(device=device, pretrained="outdoor")

    # Data
    det_ds = YOLOImagesDataset(args.det_yaml, split="train", patch=args.patch)
    mid_ds = MIDDualViewDataset(args.mid_root, K_per_pair=3, patch=args.patch)

    det_loader = make_loader(det_ds, args.batch, workers=2, shuffle=True)
    mid_loader = make_loader(mid_ds, args.batch, workers=2, shuffle=True)

    det_iter = iter(det_loader)
    mid_iter = iter(mid_loader)

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # Train
    for step in range(1, args.steps+1):
        opt.zero_grad(set_to_none=True)

        # (A) detection batch (priors only)
        try: det = next(det_iter)
        except StopIteration:
            det_iter = iter(det_loader); det = next(det_iter)
        det = det.to(device)
        if device == "cuda": det = det.half()

        with amp_autocast(device):
            enh_det = model(det)
            loss_prior = args.w_edge * edge(enh_det, det) + args.w_rc * rc(enh_det, det)
            if tv is not None:
                loss_prior = loss_prior + tv(enh_det)

        # (B) MID batch (LoFTR + priors)
        try: A, B = next(mid_iter)
        except StopIteration:
            mid_iter = iter(mid_loader); A, B = next(mid_iter)
        A = A.to(device); B = B.to(device)
        if device == "cuda": A = A.half(); B = B.half()

        with amp_autocast(device):
            enhA = model(A); enhB = model(B)
            loss_loftr = args.w_loftr * loftr(enhA.float(), enhB.float(), A.float(), B.float())  # use float in LoFTR
            loss_rc    = args.w_rc * (rc(enhA, A) + rc(enhB, B)) * 0.5
            loss_edge  = args.w_edge * (edge(enhA, A) + edge(enhB, B)) * 0.5
            loss = loss_prior + loss_loftr + loss_rc + loss_edge

        loss.backward()
        opt.step()

        if step % 50 == 0:
            # For logging parity with your earlier prints:
            # 'prior' ~ edge on det + rc on det; I report the det prior (dominant) to keep line short.
            with torch.no_grad():
                # decouple for clearer logs (approximate)
                prior_det = loss_prior.detach().float().item()
                rc_val = (rc(enhA, A) + rc(enhB, B)).detach().float().item() * 0.5
                loftr_val = loss_loftr.detach().float().item()
            print(f"[{step}/{args.steps}] L={loss.item():.4f} | prior={prior_det:.3f} rc={rc_val:.3f} loftr={loftr_val:.3f}")

        if step % 500 == 0 or step == args.steps:
            save_checkpoint(args.ckpt_out, model, extra={"step": step})

    print(f"[*] Done. Saved -> {args.ckpt_out}")

if __name__ == "__main__":
    main()
