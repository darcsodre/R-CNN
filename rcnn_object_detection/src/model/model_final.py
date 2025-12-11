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
        device: str | None = None,
    ):
        super().__init__()

        # Dispositivo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Carrega modelo base
        self.model = fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None
        )

        # === ALTERAÇÃO: congelar backbone para reduzir overfitting com pouco dado ===
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Substitui a cabeça (classification head)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes  # inclui background!
        )

        self.model.to(self.device)

        # === ALTERAÇÃO: Adam com weight_decay e apenas parâmetros treináveis ===
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

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

        # === NOVO: early stopping bem simples baseado em validation loss ===
        '''patience = 5
        best_val_loss = float("inf")
        epochs_no_improve = 0'''

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
                # IMPORTANTE: modelo fica em modo train, porque em eval()
                # o Faster R-CNN NÃO aceita targets (só imagens) para calcular loss.
                val_total = 0.0
                with torch.no_grad():
                    for images, targets in validation_data:
                        images = [img.to(self.device) for img in images]
                        targets = [
                            {k: v.to(self.device) for k, v in t.items()}
                            for t in targets
                        ]

                        loss_dict = self.model(images, targets)
                        loss = sum(loss_dict.values())
                        val_total += loss.item()

                avg_val_loss = val_total / max(1, len(validation_data))
                val_losses.append(avg_val_loss)
                LOGGER.info(
                    f">>> Epoch {epoch+1} validation average loss: {avg_val_loss:.4f}"
                )

                # === NOVO: early stopping baseado em validation loss ===
                '''if avg_val_loss < best_val_loss - 1e-4:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    LOGGER.info(
                        f"Early stopping acionado na época {epoch+1} "
                        f"(melhor val_loss = {best_val_loss:.4f})"
                    )
                    break'''

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
        images = [img.to(self.device) for img in images]

        outputs = self.model(images)

        # Filtra por score
        filtered = []
        for out in outputs:
            keep = out["scores"] >= score_thresh
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
