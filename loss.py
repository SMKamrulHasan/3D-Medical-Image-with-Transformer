# ==============================
# Losses / Metrics
# ==============================
def dice_loss_from_logits(logits, targets, num_classes, eps=1e-6):
    if num_classes == 1:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        inter = (probs * targets).sum(dim=(2,3,4))
        denom = probs.sum(dim=(2,3,4)) + targets.sum(dim=(2,3,4)) + eps
        dice = (2*inter + eps) / denom
        return 1 - dice.mean()
    else:
        probs = F.softmax(logits, dim=1)
        oh = F.one_hot(targets.long(), num_classes).permute(0,4,1,2,3).float()
        inter = (probs * oh).sum(dim=(2,3,4))
        denom = probs.sum(dim=(2,3,4)) + oh.sum(dim=(2,3,4)) + eps
        dice = (2*inter + eps) / denom
        return 1 - dice.mean()

def make_criterion(num_classes, dice_weight=0.5):
    if num_classes == 1:
        ce = nn.BCEWithLogitsLoss()
        def loss_fn(logits, targets):
            return dice_weight * dice_loss_from_logits(logits, targets, 1) + (1-dice_weight)*ce(logits, targets.float())
    else:
        ce = nn.CrossEntropyLoss()
        def loss_fn(logits, targets):
            return dice_weight * dice_loss_from_logits(logits, targets, num_classes) + (1-dice_weight)*ce(logits, targets.long())
    return loss_fn

@torch.no_grad()
def dice_metric_from_probs(probs, targets, num_classes):
    eps = 1e-6
    if num_classes == 1:
        targets = targets.float()
        inter = (probs * targets).sum(dim=(2,3,4))
        denom = probs.sum(dim=(2,3,4)) + targets.sum(dim=(2,3,4)) + eps
        dice = (2*inter + eps) / denom
        return dice.mean().item()
    else:
        oh = F.one_hot(targets.long(), probs.size(1)).permute(0,4,1,2,3).float()
        inter = (probs * oh).sum(dim=(2,3,4))
        denom = probs.sum(dim=(2,3,4)) + oh.sum(dim=(2,3,4)) + eps
        dice = (2*inter + eps) / denom
        return dice.mean().item()

