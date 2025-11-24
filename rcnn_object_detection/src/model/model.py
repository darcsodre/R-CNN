import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

LOGGER = logging.getLogger(__name__)


class FasterRCNNDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        pretrained: bool = True,
        device: str = None,
    ):
        super().__init__()

        # Dispositivo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Carrega modelo base
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)

        # Substitui a cabeça (classification head)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes  # inclui background!
        )

        self.model.to(self.device)

        # Otimizador
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_model(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        print_every: int = 5,
        validation_data: DataLoader | None = None,
    ) -> tuple[list[float], list[float] | None]:
        self.model.train()
        epoch_losses: list[float] = []
        val_losses: list[float] | None = [] if validation_data is not None else None

        for epoch in range(epochs):
            total_loss = 0.0

            for i, (images, targets) in enumerate(dataloader):
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if i % print_every == 0:
                    LOGGER.info(
                        f"[Epoch {epoch+1}/{epochs}] Step {i} Loss: {loss.item():.4f}"
                    )

            avg_loss = total_loss / max(1, len(dataloader))
            epoch_losses.append(avg_loss)
            LOGGER.info(f">>> Epoch {epoch+1} final average loss: {avg_loss:.4f}")

            # Validation pass (if provided)
            if validation_data is not None:
                self.model.eval()
                val_total = 0.0
                with torch.no_grad():
                    for images, targets in validation_data:
                        images = [img.to(self.device) for img in images]
                        targets = [
                            {k: v.to(self.device) for k, v in t.items()}
                            for t in targets
                        ]

                        # CORRETO: retorna um único dict
                        losses_dict = self.model(images, targets)
                        for loss_dict in losses_dict:

                            # CORRETO: soma os losses do dict
                            loss = sum(loss_dict.values()).item()

                        val_total += loss

                avg_val_loss = val_total / max(1, len(validation_data))
                val_losses.append(avg_val_loss)
                LOGGER.info(
                    f">>> Epoch {epoch+1} validation average loss: {avg_val_loss:.4f}"
                )

                self.model.train()

        return epoch_losses, val_losses

    @torch.no_grad()
    def predict(
        self, images: list[torch.Tensor], score_thresh: float = 0.5
    ) -> list[dict[str, torch.Tensor]]:
        """
        Predict bounding boxes for a list of images.

        Parameters
        ----------
        images : list[torch.Tensor]
            List of input images as tensors.
        score_thresh : float, optional
            Score threshold to filter boxes, by default 0.5

        Returns
        -------
        list[dict[str, torch.Tensor]]
            List of predictions, each containing 'boxes', 'labels', and 'scores'.
        """
        self.model.eval()
        images: list[torch.Tensor] = [img.to(self.device) for img in images]

        outputs = self.model(images)

        # Filtra por score
        filtered = []
        for out in outputs:
            keep: bool = out["scores"] >= score_thresh
            filtered.append(
                {
                    "boxes": out["boxes"][keep].cpu(),
                    "labels": out["labels"][keep].cpu(),
                    "scores": out["scores"][keep].cpu(),
                }
            )
        return filtered

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        LOGGER.info(f"Modelo salvo em: {path}")

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        LOGGER.info(f"Modelo carregado de: {path}")
