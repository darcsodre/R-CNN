import os
import logging
import random   # === NOVO: para embaralhar índices e fazer split 80/10/10 ===

import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from collections import Counter  # para descobrir classes automaticamente

from model.model_final import FasterRCNNDetector
import random
import numpy as np
import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


LOGGER = logging.getLogger(__name__)

# GPU / device detection and logging
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    try:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    except Exception:
        gpu_names = ["unknown"] * num_gpus
    print(f"CUDA available. Using GPU. Count: {num_gpus}. Devices: {gpu_names}")
    print(
        f"torch.version.cuda={torch.version.cuda}, cudnn_version={torch.backends.cudnn.version()}"
    )
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available. Using CPU.")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "..", "data")

# === ALTERAÇÃO: usar apenas train_data_augmentation (1200) + test_data (60) ===
AUG_DATA_PATH = os.path.join(DATA_PATH, "train_data_augmentation")
AUG_ANN_PATH = os.path.join(AUG_DATA_PATH, "Annotations")
AUG_IMG_PATH = os.path.join(AUG_DATA_PATH, "JPEGImages")

TEST_DATA_PATH = os.path.join(DATA_PATH, "test_data")
TEST_ANN_PATH = os.path.join(TEST_DATA_PATH, "Annotations")
TEST_IMG_PATH = os.path.join(TEST_DATA_PATH, "JPEGImages")


# === NOVO: Métricas de detecção (IoU, precision, recall, F1, acurácia de classe) ===
def compute_iou(boxA, boxB) -> float:
    """
    boxA e boxB no formato [xmin, ymin, xmax, ymax]
    Retorna IoU em [0, 1].
    """
    xA = max(float(boxA[0]), float(boxB[0]))
    yA = max(float(boxA[1]), float(boxB[1]))
    xB = min(float(boxA[2]), float(boxB[2]))
    yB = min(float(boxA[3]), float(boxB[3]))

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    areaA = max(0.0, float(boxA[2] - boxA[0])) * max(
        0.0, float(boxA[3] - boxA[1])
    )
    areaB = max(0.0, float(boxB[2] - boxB[0])) * max(
        0.0, float(boxB[3] - boxB[1])
    )

    union = areaA + areaB - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def evaluate_detector(
    model: FasterRCNNDetector,
    images: list[torch.Tensor],
    annotations: list[dict[str, torch.Tensor]],
    iou_thresh: float = 0.5,
    score_thresh: float = 0.7,
) -> dict[str, float]:
    """
    Avalia o modelo:
      - TP, FP, FN
      - precision, recall, F1
      - IoU médio dos matches corretos
      - acurácia de classe (rótulo certo dado que o box encostou)
    Considera TP quando:
      IoU >= iou_thresh E label correto.
    """
    model.eval()
    #preds = model.predict(images, score_thresh=score_thresh)
    batch_eval = 2   # pode trocar para 1 caso ainda dê out-of-memory
    all_preds = []

    for i in range(0, len(images), batch_eval):
        batch_imgs = images[i : i + batch_eval]
        batch_preds = model.predict(batch_imgs, score_thresh=score_thresh)
        all_preds.extend(batch_preds)

    preds = all_preds

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_matched_iou = 0
    total_cls_correct = 0
    ious_correct: list[float] = []

    for pred, target in zip(preds, annotations):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        true_boxes = target["boxes"]
        true_labels = target["labels"]

        total_gt += len(true_boxes)
        total_pred += len(pred_boxes)

        used_true = set()

        for pb, pl in zip(pred_boxes, pred_labels):
            best_iou = 0.0
            best_idx = -1
            best_true_label = None

            for idx, (tb, tl) in enumerate(zip(true_boxes, true_labels)):
                if idx in used_true:
                    continue
                iou = compute_iou(pb, tb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    best_true_label = int(tl)

            if best_idx >= 0 and best_iou >= iou_thresh:
                total_matched_iou += 1
                used_true.add(best_idx)

                if int(pl) == best_true_label:
                    total_cls_correct += 1
                    total_tp += 1
                    ious_correct.append(best_iou)
                else:
                    # box em cima, mas classe errada
                    pass
            else:
                # falso positivo geométrico (sem GT correspondente)
                pass

    fp = total_pred - total_tp
    fn = total_gt - total_tp

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(ious_correct) / len(ious_correct) if ious_correct else 0.0
    cls_acc = (
        total_cls_correct / total_matched_iou if total_matched_iou > 0 else 0.0
    )

    return {
        "total_gt": float(total_gt),
        "total_pred": float(total_pred),
        "tp": float(total_tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou),
        "cls_acc": float(cls_acc),
    }


# Função para descobrir classes automaticamente nos XML (igual à sua)
def discover_classes(
    annotations_dirs: list[str],
) -> tuple[dict[str, int], dict[int, str], Counter]:
    """
    Varre pastas de anotações (.xml), descobre os valores de <name>
    e monta o mapeamento classe->id e id->classe.
    """
    class_counter: Counter = Counter()

    for ann_dir in annotations_dirs:
        if not os.path.isdir(ann_dir):
            continue
        for xml_file in os.listdir(ann_dir):
            if not xml_file.endswith(".xml"):
                continue
            xml_path = os.path.join(ann_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                LOGGER.warning(f"Erro ao ler {xml_path}: {e}")
                continue

            for obj in root.iter("object"):
                name_tag = obj.find("name")
                if name_tag is not None and name_tag.text:
                    class_counter[name_tag.text] += 1

    if not class_counter:
        raise RuntimeError("Nenhuma classe encontrada nas anotações XML.")

    class_names = sorted(class_counter.keys())
    class_map = {name: i + 1 for i, name in enumerate(class_names)}  # 0 = background
    id_to_name = {i + 1: name for i, name in enumerate(class_names)}

    print("\n=== CLASSES ENCONTRADAS NOS XML ===")
    for name in class_names:
        print(f"  id={class_map[name]}  classe='{name}'  (n={class_counter[name]})")
    print("===================================\n")

    return class_map, id_to_name, class_counter


def plot_losses(train_loss: list[float], val_loss: list[float] | None = None) -> None:
    fig = plt.figure(dpi=160, figsize=(9, 6))
    ax = fig.gca()
    ax.plot(range(1, len(train_loss) + 1), train_loss, color="blue", marker="o")
    if val_loss is not None:
        ax.plot(range(1, len(val_loss) + 1), val_loss, color="purple", marker="o")
        ax.legend(["Training Loss", "Validation Loss"])
    else:
        ax.legend(["Training Loss"])
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.grid()
    plt.show()


# Função de plot agora aceita labels previstos e nomes de classe (igual à sua, só mudei cor do texto)
def plot_image_with_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    title: str = "Image with Bounding Boxes",
    true_boxes: torch.Tensor | None = None,
    pred_labels: torch.Tensor | None = None,
    class_id_to_name: dict[int, str] | None = None,
    pdf: PdfPages | None = None,
) -> None:
    pil_image = torchvision.transforms.functional.to_pil_image(image)
    draw = ImageDraw.Draw(pil_image)

    # Caixas previstas (vermelho + nome da classe)
    for i, box in enumerate(boxes):
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline="red",
            width=2,
        )
        if pred_labels is not None and class_id_to_name is not None and i < len(
            pred_labels
        ):
            cls_id = int(pred_labels[i])
            cls_name = class_id_to_name.get(cls_id, str(cls_id))
            draw.text((box[0] + 2, box[1] + 2), cls_name, fill="purple")

    # Caixas verdadeiras (verde)
    if true_boxes is not None:
        for box in true_boxes:
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])],
                outline="green",
                width=2,
            )

    fig = plt.figure(dpi=160, figsize=(9, 6))
    plt.imshow(pil_image)
    plt.title(title)
    plt.axis("off")

    if pdf is not None:
        pdf.savefig(fig)

    plt.show()
    plt.close(fig)


def load_data_set(
    index_file_path: str,
    jpeg_path: str,
    annotations_path: str,
    class_map: dict[str, int],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    map_image_name_to_xml: dict[str, str] = {}
    with open(index_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        image_name, xml_name = line.split(" ")
        image_name = image_name.split("/")[-1]
        xml_name = xml_name.replace("\n", "").split("/")[-1]
        map_image_name_to_xml[image_name] = xml_name
    LOGGER.info(f"Number of images to train: {len(map_image_name_to_xml)}")

    input_images: list[torch.Tensor] = []
    output_annotations: list[dict[str, torch.Tensor]] = []

    for image_name, xml_name in map_image_name_to_xml.items():
        image_path = os.path.join(jpeg_path, image_name)
        xml_path = os.path.join(annotations_path, xml_name)

        image = Image.open(image_path).convert("RGB")
        input_images.append(torchvision.transforms.functional.to_tensor(image))

        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes: list[list[int]] = []
        labels: list[int] = []

        for obj in root.iter("object"):
            name_tag = obj.find("name")
            bnd = obj.find("bndbox")
            if name_tag is None or bnd is None:
                continue

            class_name = name_tag.text
            if class_name not in class_map:
                LOGGER.warning(f"Classe desconhecida '{class_name}' em {xml_path}")
                continue

            xmin = int(bnd.find("xmin").text)
            ymin = int(bnd.find("ymin").text)
            xmax = int(bnd.find("xmax").text)
            ymax = int(bnd.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_map[class_name])

        if not boxes:
            LOGGER.warning(f"Imagem {image_name} sem boxes válidos, ignorando.")
            input_images.pop()
            continue

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        annotation = {"boxes": boxes_tensor, "labels": labels_tensor}
        output_annotations.append(annotation)

    return input_images, output_annotations


def main():
    # === ALTERAÇÃO: descobrir classes usando augmentation + test ===
    CLASS_MAP, ID_TO_NAME, class_counts = discover_classes(
        [
            AUG_ANN_PATH,
            TEST_ANN_PATH,
        ]
    )

    # === Carregar 1200 (augmentation) + 60 (test) e dividir 80/10/10 ===
    LOGGER.info("Loading AUGMENTED (1200) data...")
    aug_index_file_path = os.path.join(AUG_DATA_PATH, "index.txt")
    aug_images, aug_annotations = load_data_set(
        aug_index_file_path, AUG_IMG_PATH, AUG_ANN_PATH, CLASS_MAP
    )

    LOGGER.info("Loading TEST (60) data...")
    test_index_file_path = os.path.join(TEST_DATA_PATH, "index.txt")
    test_images, test_annotations = load_data_set(
        test_index_file_path, TEST_IMG_PATH, TEST_ANN_PATH, CLASS_MAP
    )

    all_images = aug_images + test_images
    all_annotations = aug_annotations + test_annotations
    total_samples = len(all_images)
    LOGGER.info(f"Total de amostras disponíveis (aug+test): {total_samples}")

    indices = list(range(total_samples))
    random.shuffle(indices)

    train_ratio = 0.8
    val_ratio = 0.1
    train_end = int(train_ratio * total_samples)
    val_end = int((train_ratio + val_ratio) * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    input_images = [all_images[i] for i in train_indices]
    output_annotations = [all_annotations[i] for i in train_indices]

    val_input_images = [all_images[i] for i in val_indices]
    val_output_annotations = [all_annotations[i] for i in val_indices]

    test_input_images = [all_images[i] for i in test_indices]
    test_output_annotations = [all_annotations[i] for i in test_indices]

    print("Total de imagens para TREINO:", len(input_images))
    print("Total de imagens para VALIDAÇÃO:", len(val_input_images))
    print("Total de imagens para TESTE:", len(test_input_images))

    dataset = list(zip(input_images, output_annotations))
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda x: list(zip(*x))
    )

    val_dataset = list(zip(val_input_images, val_output_annotations))
    val_dataloader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: list(zip(*x))
    )

    LOGGER.info("Initializing model...")
    num_classes = len(CLASS_MAP) + 1
    model = FasterRCNNDetector(num_classes=num_classes, device=DEVICE, lr=5e-5)

    LOGGER.info("Starting training...")
    train_loss, val_loss = model.train_model(
        dataloader,
        epochs=30,
        print_every=25,
        validation_data=val_dataloader,
    )
    LOGGER.info("Training completed.")
    LOGGER.info("Saving model...")
    model_save_path = os.path.join(BASE_PATH, "fasterrcnn_model.pth")
    model.save(model_save_path)

    plot_losses(train_loss, val_loss)

    print("\n>>> Iniciando cálculo das métricas (validação e teste). "
          "Isso pode levar alguns segundos...\n")

    # === NOVO: métricas na VALIDAÇÃO ===
    metrics_val = evaluate_detector(
        model,
        val_input_images,
        val_output_annotations,
        iou_thresh=0.5,
        score_thresh=0.9,
    )

    print("\n=== Métricas de detecção na VALIDAÇÃO ===")

    print(
        f"Total de defeitos anotados nas imagens de validação (GT): "
        f"{metrics_val['total_gt']:.0f}"
    )

    print(
        f"Total de defeitos detectados pelo modelo (Predições): "
        f"{metrics_val['total_pred']:.0f}"
    )

    print(
        f"Verdadeiros Positivos (TP) – defeitos corretamente detectados: "
        f"{metrics_val['tp']:.0f}"
    )

    print(
        f"Falsos Positivos (FP) – detecções incorretas (alarmes falsos): "
        f"{metrics_val['fp']:.0f}"
    )

    print(
        f"Falsos Negativos (FN) – defeitos reais não detectados pelo modelo: "
        f"{metrics_val['fn']:.0f}"
    )

    print(
        f"Precisão (Precision) – proporção de detecções corretas entre todas as detecções: "
        f"{metrics_val['precision']:.3f}"
    )

    print(
        f"Revocação (Recall) – proporção de defeitos reais que foram detectados: "
        f"{metrics_val['recall']:.3f}"
    )

    print(
        f"F1-score – média harmônica entre precisão e recall: "
        f"{metrics_val['f1']:.3f}"
    )

    print(
        f"IoU médio – sobreposição média entre as caixas previstas e as caixas reais: "
        f"{metrics_val['mean_iou']:.3f}"
    )

    print(
        f"Acurácia de classe – proporção de classes corretamente previstas (quando há match): "
        f"{metrics_val['cls_acc']:.3f}"
    )


    '''print("\n=== Métricas de detecção na VALIDAÇÃO ===")
    print(f"Total GT:   {metrics_val['total_gt']:.0f}")
    print(f"Total pred: {metrics_val['total_pred']:.0f}")
    print(f"TP:         {metrics_val['tp']:.0f}")
    print(f"FP:         {metrics_val['fp']:.0f}")
    print(f"FN:         {metrics_val['fn']:.0f}")
    print(f"Precision:  {metrics_val['precision']:.3f}")
    print(f"Recall:     {metrics_val['recall']:.3f}")
    print(f"F1-score:   {metrics_val['f1']:.3f}")
    print(f"IoU médio:  {metrics_val['mean_iou']:.3f}")
    print(f"Acurácia de classe: {metrics_val['cls_acc']:.3f}")'''

   
    # === NOVO: métricas no TESTE (10%) ===
    metrics_test = evaluate_detector(
        model,
        test_input_images,
        test_output_annotations,
        iou_thresh=0.5,
        score_thresh=0.7,
    )

    print("\n=== Métricas de detecção no TESTE ===")

    print(
        f"Total de defeitos anotados nas imagens de teste (GT): "
        f"{metrics_test['total_gt']:.0f}"
    )

    print(
        f"Total de defeitos detectados pelo modelo (Predições): "
        f"{metrics_test['total_pred']:.0f}"
    )

    print(
        f"Verdadeiros Positivos (TP) – defeitos corretamente detectados: "
        f"{metrics_test['tp']:.0f}"
    )

    print(
        f"Falsos Positivos (FP) – detecções incorretas (alarmes falsos): "
        f"{metrics_test['fp']:.0f}"
    )

    print(
        f"Falsos Negativos (FN) – defeitos reais não detectados pelo modelo: "
        f"{metrics_test['fn']:.0f}"
    )

    print(
        f"Precisão (Precision) – proporção de detecções corretas entre todas as detecções: "
        f"{metrics_test['precision']:.3f}"
    )

    print(
        f"Revocação (Recall) – proporção de defeitos reais que foram detectados: "
        f"{metrics_test['recall']:.3f}"
    )

    print(
        f"F1-score – média harmônica entre precisão e recall: "
        f"{metrics_test['f1']:.3f}"
    )

    print(
        f"IoU médio – sobreposição média entre as caixas previstas e as caixas reais: "
        f"{metrics_test['mean_iou']:.3f}"
    )

    print(
        f"Acurácia de classe – proporção de classes corretamente previstas (quando há match): "
        f"{metrics_test['cls_acc']:.3f}"
    )

    '''print("\n=== Métricas de detecção no TESTE ===")
    print(f"Total GT:   {metrics_test['total_gt']:.0f}")
    print(f"Total pred: {metrics_test['total_pred']:.0f}")
    print(f"TP:         {metrics_test['tp']:.0f}")
    print(f"FP:         {metrics_test['fp']:.0f}")
    print(f"FN:         {metrics_test['fn']:.0f}")
    print(f"Precision:  {metrics_test['precision']:.3f}")
    print(f"Recall:     {metrics_test['recall']:.3f}")
    print(f"F1-score:   {metrics_test['f1']:.3f}")
    print(f"IoU médio:  {metrics_test['mean_iou']:.3f}")
    print(f"Acurácia de classe: {metrics_test['cls_acc']:.3f}")'''


    print("\n>>> Avaliação concluída. Agora vou perguntar "
          "quantas imagens de validação você quer visualizar.\n")

    # === SUA PARTE ANTIGA: escolher quantas imagens de validação visualizar ===
    try:
        n_to_show = int(
            input(
                "Quantas imagens de validação deseja visualizar? "
                "(0 = nenhuma, -1 = todas): "
            )
        )
    except ValueError:
        n_to_show = 0

    if n_to_show != 0:
        if n_to_show < 0 or n_to_show > len(val_input_images):
            n_to_show = len(val_input_images)

        LOGGER.info(f"Plotando {n_to_show} imagens de validação com predições...")
        prediction_list = model.predict(
            images=val_input_images[:n_to_show], score_thresh=0.7
        )

        pdf_path = os.path.join(BASE_PATH, "val_predictions.pdf")

        with PdfPages(pdf_path) as pdf:
            for i, (prediction, true_annotation) in enumerate(
                zip(prediction_list, val_output_annotations[:n_to_show])
            ):
                boxes = prediction["boxes"]
                scores = prediction["scores"]
                labels = prediction["labels"]

                LOGGER.info(f"Imagem {i} - boxes: {boxes}")
                LOGGER.info(f"Imagem {i} - scores: {scores}")
                LOGGER.info(f"Imagem {i} - labels (ids): {labels}")

                plot_image_with_boxes(
                    val_input_images[i],
                    boxes,
                    title=f"Validation Image {i} with Predicted Boxes",
                    true_boxes=true_annotation["boxes"],
                    pred_labels=labels,
                    class_id_to_name=ID_TO_NAME,
                    pdf=pdf,
                )

        print(f"\nPDF gerado com {n_to_show} imagens: {pdf_path}\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt=r"%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
