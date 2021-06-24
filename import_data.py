import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class chicken_data(Dataset):

    def __init__(self, img_dir = "data/annotation/images", annotations_dir = "data/annotation/xml", transform = None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

        self.img_names = os.listdir(img_dir)
        self.img_names.sort()
        self.img_names = [os.path.join(img_dir, img_name) for img_name in self.img_names]

        self.annotation_names = os.listdir(annotations_dir)
        self.annotation_names.sort()
        self.annotation_names = [os.path.join(annotations_dir, ann_name) for ann_name in self.annotation_names]

    def __getitem__(self, index):
        img_names = self.img_names[index]
        img = Image.open(img_names)

        annotation_name = self.annotation_names[index]
        annotation_tree = ET.parse(annotation_name) # XML parser to load the file
        object_xmls = annotation_tree.findall("object")

        image_id = str(annotation_tree.find("filename").text)

        size_xml = annotation_tree.find("size")
        img_width = str(size_xml.find("width").text)
        img_height = str(size_xml.find("height").text)
        img_depth = str(size_xml.find("depth").text)
        image_size = [img_width, img_height, img_depth]

        boxes = []
        labels = []
        difficult = []

        for object_xml in object_xmls:

            bndbox_xml = object_xml.find("bndbox")

            label = str(object_xml.find("name").text)
            labels.append(label)

            diff = int(object_xml.find("difficult").text)
            difficult.append(diff)

            xmax = int(bndbox_xml.find("xmax").text)
            xmin = int(bndbox_xml.find("xmin").text)
            ymax = int(bndbox_xml.find("ymax").text)
            ymin = int(bndbox_xml.find("ymin").text)

            # Convert coordinates for the boxes
            w = xmax - xmin
            h = ymax - ymin
            x = int(xmin + w/2)
            y = int(ymin + h/2)

            # Normalize
            x /= img.size[0]
            w /= img.size[0]
            y /= img.size[1]
            h /= img.size[1]

            bndbox = (x, y, w, h)
            bndbox = torch.tensor(bndbox)
            boxes.append(bndbox)
        
        if self.transform:
                img = self.transform(img)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["img_id"] = image_id
        target["img_size"] = image_size
        target["difficult"] = difficult

        return img, target

    def __len__(self):
        return len(self.img_names)

def unpack_bndbox(bndbox, img):
    x, y, w, h = tuple(bndbox)
    x *= img.size[0]
    w *= img.size[0]
    y *= img.size[1]
    h *= img.size[1]

    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2
    bndbox = [xmin, ymin, xmax, ymax]
    return bndbox

def show(batch, pred_bndbox = None):
    img, target = batch
    boxes = target["boxes"]
    
    # Draw image
    img = transforms.ToPILImage()(img)
    #img = transforms.Resize((512, 512))(img)
    draw = ImageDraw.Draw(img)

    # Draw boxes
    for box in boxes:
        box = unpack_bndbox(box, img)
        draw.rectangle(box)
    
    # Draw prediction boxes
    if pred_bndbox is not None:
        for pred_box in pred_bndbox:
            pred_box = unpack_bndbox(pred_box, img)
            draw.rectangle(pred_box, outline = 1000)
    
    img.show()

if __name__ == "__main__":
    dataset = chicken_data(transform=transforms.ToTensor())
    print("len dataset: ", len(dataset))

    exs = dataset[0]
    show(exs)