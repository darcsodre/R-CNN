import os

import logging

import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

from model.model import FasterRCNNDetector

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
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train_data")
TRAIN_ANNOTATIONS_PATH = os.path.join(TRAIN_DATA_PATH, "Annotations")
JPEG_IMAGES_PATH = os.path.join(TRAIN_DATA_PATH, "JPEGImages")
TEST_DATA_PATH = os.path.join(DATA_PATH, "test_data")
TEST_ANNOTATIONS_PATH = os.path.join(TEST_DATA_PATH, "Annotations")
JPEG_TEST_IMAGES_PATH = os.path.join(TEST_DATA_PATH, "JPEGImages")


def plot_losses(train_loss: list[float], val_loss: list[float] | None = None) -> None:
    fig = plt.figure(dpi=160, figsize=(9, 6))
    ax = fig.gca()
    ax.plot(range(1, len(train_loss) + 1), train_loss, color="blue", marker="o")
    if val_loss is not None:
        ax.plot(range(1, len(val_loss) + 1), val_loss, color="orange", marker="o")
        ax.legend(["Training Loss", "Validation Loss"])
    else:
        ax.legend(["Training Loss"])
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.grid()
    plt.show()


def plot_image_with_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    title: str = "Image with Bounding Boxes",
    true_boxes: torch.Tensor | None = None,
) -> None:
    pil_image = torchvision.transforms.functional.to_pil_image(image)
    draw = ImageDraw.Draw(pil_image)
    for box in boxes:
        draw.rectangle(
            [(box[0], box[1]), (box[2], box[3])],
            outline="red",
            width=2,
        )
    if true_boxes is not None:
        for box in true_boxes:
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])],
                outline="green",
                width=2,
            )
    plt.figure(dpi=160, figsize=(9, 6))
    plt.imshow(pil_image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_data_set(
    train_index_file_path: str, jpeg_path: str, annotations_path: str
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:

    map_image_name_to_xml: dict[str, str] = dict()
    with open(train_index_file_path, "r") as f:
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

        # Carrega imagem
        image = Image.open(image_path).convert("RGB")
        input_images.append(torchvision.transforms.functional.to_tensor(image))

        # Carrega anotação
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        for neighbor in root.iter("bndbox"):
            xmin = int(neighbor.find("xmin").text)
            ymin = int(neighbor.find("ymin").text)
            xmax = int(neighbor.find("xmax").text)
            ymax = int(neighbor.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Apenas uma classe de objeto
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        annotation = {"boxes": boxes_tensor, "labels": labels_tensor}
        output_annotations.append(annotation)
    return input_images, output_annotations


def main():
    LOGGER.info("Loading training data...")
    train_index_file_path = os.path.join(
        TRAIN_DATA_PATH, "index.txt"
    )  # Map JPEG to xml files
    input_images, output_annotations = load_data_set(
        train_index_file_path, JPEG_IMAGES_PATH, TRAIN_ANNOTATIONS_PATH
    )
    LOGGER.info("Training data loaded.")
    LOGGER.info("Loading validation data...")
    test_index_file_path = os.path.join(
        TEST_DATA_PATH, "index.txt"
    )  # Map JPEG to xml files
    val_input_images, val_output_annotations = load_data_set(
        test_index_file_path, JPEG_TEST_IMAGES_PATH, TEST_ANNOTATIONS_PATH
    )
    # LOGGER.info("Plotting sample image with boxes...")
    # plot_image_with_boxes(
    #     input_images[0], output_annotations[0]["boxes"], title="Sample Training Image"
    # )
    dataset = list(zip(input_images, output_annotations))
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda x: list(zip(*x))
    )
    test_dataset = list(zip(val_input_images, val_output_annotations))
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: list(zip(*x)),
    )
    LOGGER.info("Initializing model...")
    model = FasterRCNNDetector(num_classes=2, device=DEVICE, lr=1e-4)
    LOGGER.info("Starting training...")
    train_loss, val_loss = model.train_model(
        dataloader,
        epochs=35,
        print_every=25,
    )
    LOGGER.info("Training completed.")
    LOGGER.info("Saving model...")
    model_save_path = os.path.join(BASE_PATH, "fasterrcnn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    plot_losses(train_loss, val_loss)
    LOGGER.info("Plotting sample validation image with predicted boxes...")
    prediction_list = model.predict(images=val_input_images[:1], score_thresh=0.5)
    for prediction, true_annotation in zip(prediction_list, val_output_annotations[:1]):
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        LOGGER.info(f"Predicted boxes: {boxes}")
        LOGGER.info(f"Scores: {scores}")
        plot_image_with_boxes(
            val_input_images[0],
            boxes,
            title="Sample Validation Image with Predicted Boxes",
            true_boxes=true_annotation["boxes"],
        )

    print()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt=r"%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
