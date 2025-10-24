# ==============================
# NIfTI Dataset
# ==============================
def _load_nifti(path: str) -> np.ndarray:
    arr = np.asanyarray(nib.load(path).dataobj)  # lazy load → numpy
    # Ensure shape is [D,H,W] (nib is usually [X,Y,Z] := [W,H,D]); we’ll standardize to [D,H,W]
    if arr.ndim == 4:
        # if multi-channel 4D (e.g., T1/T2/...), take channel 0 or mean — adjust to your need
        arr = arr[..., 0]
    # We assume input order is (X,Y,Z). Convert to (Z,Y,X) = (D,H,W)
    arr = np.moveaxis(arr, (0,1,2), (2,1,0))
    return arr

def _resize_volume_torch(vol: torch.Tensor, out_shape: Tuple[int,int,int], mode: str) -> torch.Tensor:
    # vol: [1,1,D,H,W] -> resize to out_shape using trilinear/nearest
    return F.interpolate(vol, size=out_shape, mode=mode, align_corners=False if mode=="trilinear" else None)

class NiftiSeg3D(Dataset):
    """
    Expects parallel NIfTI files in two folders:
      vols_dir : *.nii or *.nii.gz (float scans)
      masks_dir: *.nii or *.nii.gz (int labels; 0..K-1 or {0,1})
    Will resize to target_shape (D,H,W) (default 16x96x96).
    """
    def __init__(self, vols_dir, masks_dir, target_shape=(16,96,96),
                 augment=True, num_classes=2, binarize_mask=False):
        self.vol_paths  = sorted(sum([glob.glob(os.path.join(vols_dir, p)) for p in ("*.nii", "*.nii.gz")], []))
        self.mask_paths = sorted(sum([glob.glob(os.path.join(masks_dir, p)) for p in ("*.nii", "*.nii.gz")], []))
        assert len(self.vol_paths) == len(self.mask_paths) and len(self.vol_paths) > 0
        self.target_shape = target_shape
        self.augment = augment
        self.num_classes = num_classes
        self.binarize_mask = binarize_mask

        # name match (basename without extensions)
        def stem(p):
            n = os.path.basename(p)
            return n.replace(".nii.gz","").replace(".nii","")
        for v, m in zip(self.vol_paths, self.mask_paths):
            assert stem(v) == stem(m), f"Name mismatch: {v} vs {m}"

    def __len__(self): return len(self.vol_paths)

    def _random_flip(self, v, m):
        if random.random() < 0.5: v = v.flip(2); m = m.flip(2)   # D
        if random.random() < 0.5: v = v.flip(3); m = m.flip(3)   # H
        if random.random() < 0.5: v = v.flip(4); m = m.flip(4)   # W
        return v, m

    def __getitem__(self, idx):
        # Load
        v_np = _load_nifti(self.vol_paths[idx]).astype(np.float32)  # [D,H,W]
        m_np = _load_nifti(self.mask_paths[idx])
        # normalize z-score per volume
        v_np = (v_np - v_np.mean()) / (v_np.std() + 1e-8)

        # torch tensors
        v = torch.from_numpy(v_np)[None,None,...]   # [1,1,D,H,W]
        m = torch.from_numpy(m_np).long()[None,None,...]  # [1,1,D,H,W]

        # resize to target (image: trilinear; mask: nearest)
        v = _resize_volume_torch(v, self.target_shape, mode="trilinear")
        m = _resize_volume_torch(m.float(), self.target_shape, mode="nearest").long()

        # basic aug
        if self.augment:
            v, m = self._random_flip(v, m)

        v = v.squeeze(0)          # [1,D,H,W]
        if self.num_classes == 1:
            if self.binarize_mask:
                m = (m > 0).float()
            else:
                m = (m.clamp_min(0).clamp_max(1)).float()  # ensure 0/1
            m = m.squeeze(0)        # [1,D,H,W]
        else:
            m = m.squeeze(0).squeeze(0)  # [D,H,W] int labels

        return v, m


# ==============================
# Train / Val loops (unchanged)
# ==============================
def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, num_classes, grad_clip=None):
    model.train()
    running = 0.0
    for v, m in loader:
        v = v.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(v)
            loss = loss_fn(logits, m)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        running += loss.item() * v.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, num_classes):
    model.eval()
    losses, dices = [], []
    ce = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    for v, m in loader:
        v = v.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        logits = model(v)
        if num_classes == 1:
            dl = dice_loss_from_logits(logits, m, 1)
            l = 0.5*dl + 0.5*ce(logits, m.float())
            probs = torch.sigmoid(logits)
        else:
            dl = dice_loss_from_logits(logits, m, num_classes)
            l = 0.5*dl + 0.5*ce(logits, m.long())
            probs = F.softmax(logits, dim=1)
        losses.append(l.item())
        dices.append(dice_metric_from_probs(probs, m, num_classes))
    return float(np.mean(losses)), float(np.mean(dices))


# ==============================
# Main
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_vols", type=str, required=True, help="dir with *.nii/.nii.gz volumes")
    ap.add_argument("--train_masks", type=str, required=True, help="dir with *.nii/.nii.gz masks")
    ap.add_argument("--val_vols", type=str, required=True)
    ap.add_argument("--val_masks", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./checkpoints_unetr_nii")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--num_classes", type=int, default=2, help="1=binary with sigmoid, >=2=softmax")
    ap.add_argument("--target_d", type=int, default=16)
    ap.add_argument("--target_h", type=int, default=96)
    ap.add_argument("--target_w", type=int, default=96)
    ap.add_argument("--binarize_mask", action="store_true", help="for binary tasks if masks contain labels >1")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    target_shape = (args.target_d, args.target_h, args.target_w)

    # Datasets / Loaders
    train_ds = NiftiSeg3D(args.train_vols, args.train_masks,
                          target_shape=target_shape,
                          augment=True,
                          num_classes=args.num_classes,
                          binarize_mask=args.binarize_mask)
    val_ds   = NiftiSeg3D(args.val_vols, args.val_masks,
                          target_shape=target_shape,
                          augment=False,
                          num_classes=args.num_classes,
                          binarize_mask=args.binarize_mask)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    model = UNETR_PaperLike(
        img_size=target_shape,
        patch_size=16,
        in_channels=1,
        out_channels=(1 if args.num_classes == 1 else args.num_classes),
        embed_dim=768, depth=12, num_heads=12
    ).to(device)

    # Optimizer / Scheduler / Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    loss_fn = make_criterion(args.num_classes, dice_weight=0.5)

    best_val_dice = -1.0
    history = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, args.num_classes, grad_clip=args.grad_clip)
        va_loss, va_dice = validate(model, val_loader, device, args.num_classes)
        scheduler.step()

        print(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_dice={va_dice:.4f}  lr={scheduler.get_last_lr()[0]:.6e}  ({time.time()-t0:.1f}s)")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_dice": va_dice})

        # save latest
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "args": vars(args),
            "history": history,
        }, os.path.join(args.out_dir, "last.pt"))

        # save best
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"  ✔ Saved new best (Dice={best_val_dice:.4f})")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Done. Best val Dice:", best_val_dice)


if __name__ == "__main__":
    main()
#
#python train_unetr_3d_nii.py \
#  --train_vols /path/to/train/vols \
#  --train_masks /path/to/train/masks \
#  --val_vols   /path/to/val/vols \
#  --val_masks  /path/to/val/masks \
#  --out_dir runs/unetr_nii \
#  --epochs 200 --batch_size 2 --lr 1e-4 --num_classes 1 \
#  --target_d 16 --target_h 96 --target_w 96 \
#  --binarize_mask
