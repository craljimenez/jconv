import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

import os
from PIL import Image
from torch.utils.data import Dataset

import cv2
import os
import glob

from torchvision.transforms import v2
from torchvision import tv_tensors  # Necesario para envolver imagen y máscara
import numpy as np



class PetDatasetTransforms:
    """Transforms for the Oxford-IIIT Pet Dataset."""
    def __init__(self, size=256):
        self.image_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __call__(self, img, mask):
        img = self.image_transform(img)
        mask = self.mask_transform(mask)
        
        # Convert mask to tensor and remap class values
        # Original: 1: Pet, 2: Border, 3: Background
        # New:      0: Background, 1: Pet, 2: Border
        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask == 3] = 0 # Background
        # Pet (1) and Border (2) keep their values
        
        return img, mask


class PetDatasetWrapper(OxfordIIITPet):
    """A wrapper for the OxfordIIITPet dataset to apply transforms to both image and mask."""
    def __init__(self, root, split, transform=None, download=False):
        super().__init__(root=root, split=split, target_types='segmentation', download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask


class PetClassificationTransforms:
    """Transforms for Oxford-IIIT Pet classification."""
    def __init__(self, size=256):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img)


class PetClassificationWrapper(OxfordIIITPet):
    """Wrapper for Oxford-IIIT Pet classification setting."""
    def __init__(self, root, split, transform=None, download=False):
        super().__init__(root=root, split=split, target_types='category', download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if self.transform:
            img = self.transform(img)
        target = torch.as_tensor(target, dtype=torch.long)
        return img, target


class FolderDatasetTransforms:
    """
    Image transformations for folder-based datasets where each sub-folder corresponds to a class.

    Examples
    --------
    dataset_root/
        Healthy/
            img1.png
            ...
        Mosaic/
            ...
    """
    def __init__(self, size=256):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img)

class FolderDatasetWrapper(Dataset):
    """
    A custom PyTorch Dataset for image classification datasets 
    where sub-folders represent class labels.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initializes the Dataset. This function scans the directory 
        and builds a list of (image_path, label_index) tuples.
        
        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied 
                                            on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # --- 1. Scan and Map Classes (Done ONCE at initialization) ---
        print(f"Scanning dataset in: {root_dir}")
        classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        
        for i, class_name in enumerate(classes):
            self.class_to_idx[class_name] = i
            class_path = os.path.join(root_dir, class_name)
            
            # Find all image files in the sub-folder
            sub_folders = os.listdir(class_path)
            for filename in sub_folders:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(class_path, filename)
                    self.image_paths.append(path)
                    self.labels.append(i) # Store the integer index
        self.classes = classes
        print(f"Found {len(self.image_paths)} images across {len(classes)} classes.")


        


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)


    def __getitem__(self, idx):
        """
        Loads and returns one sample (image and label) from the dataset.
        This is the method responsible for LAZY LOADING.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            tuple: (image, label) where image is a Tensor and label is an integer.
        """
        # --- 2. Load Image Data (Done ON-DEMAND) ---
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # --- 3. Apply Transformations ---
        if self.transform:
            image = self.transform(image)

        return image, label




class YOLOSegDataset(Dataset):
    def __init__(self,
                 images_dir,
                 labels_dir,
                 transform=None,
                 classes=None,
                 target_class_id=None,
                 add_background=False):
        """
        Args:
            classes (list): Lista de IDs de clases (ej. [0, 4, 7]).
            target_class_id (int): Si se especifica, filtra solo esta clase.
            add_background (bool): 
                - Si es True, agrega un canal extra al principio (índice 0) que representa el fondo.
                - El fondo se define como todo pixel que NO pertenezca a las clases seleccionadas.
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.classes = classes
        self.target_class_id = target_class_id
        self.add_background = add_background # <--- NUEVO PARAMETRO
        
        if self.target_class_id is not None:
            # Caso A: Solo una clase objetivo
            # Si hay fondo: 2 canales (Fondo, Target)
            # Si no hay fondo: 1 canal (Target)
            self.num_classes = 1 + (1 if add_background else 0)
            
        elif self.classes is not None:
            # Caso B: Lista de clases específicas
            # Canales = Cantidad de clases + 1 (si hay fondo)
            self.num_classes = len(self.classes) + (1 if add_background else 0)
            
        else:
            # Caso C: Sin filtros (devuelve índices crudos o todas las clases)
            # Aquí es difícil saberlo sin leer los archivos, se deja en None o manual.
            self.num_classes = None

        self.image_files = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg")) + 
            glob.glob(os.path.join(images_dir, "*.png")) +
            glob.glob(os.path.join(images_dir, "*.jpeg"))
        )
    def change_num_classes(self,mask):
        self.num_classes = mask.shape[0] if mask.ndim == 3 else None
        print("Numero de clases actualizado a:", self.num_classes)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- 1. Lectura Imagen ---
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # --- 2. Preparar Máscara Temporal ---
        # IMPORTANTE: Inicializamos en -1 porque la clase 0 existe en YOLO.
        # -1 representará "Fondo vacío" por ahora.
        temp_mask = np.full((h, w), -1, dtype=np.int32)
        
        file_name = os.path.basename(img_path).rsplit('.', 1)[0]
        label_path = os.path.join(self.labels_dir, f"{file_name}.txt")

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if not data: continue
                class_id = int(data[0])
                coords = np.array([float(x) for x in data[1:]])
                
                # Filtros
                if self.target_class_id is not None:
                    if class_id != self.target_class_id: continue
                if self.classes and class_id not in self.classes: continue

                # Dibujar
                if len(coords) == 4: # Box
                    xc, yc, bw, bh = coords
                    x1, y1 = int((xc-bw/2)*w), int((yc-bh/2)*h)
                    x2, y2 = int((xc+bw/2)*w), int((yc+bh/2)*h)
                    cv2.rectangle(temp_mask, (x1, y1), (x2, y2), class_id, -1)
                elif len(coords) > 4: # Poly
                    points = coords.reshape(-1, 2)
                    points[:,0]*=w; points[:,1]*=h
                    points = points.astype(np.int32)
                    cv2.fillPoly(temp_mask, [points], class_id)

        # --- 3. Procesar Salida de Máscara (LOGICA NUEVA) ---
        
        if self.target_class_id is not None:
            # === CASO A: Target Específico ===
            target_mask = (temp_mask == self.target_class_id).astype(np.uint8)
            
            if self.add_background:
                # Si pedimos fondo: [Fondo, Target] -> (2, H, W)
                bg_mask = 1 - target_mask
                final_mask = np.stack([bg_mask, target_mask], axis=0)
            else:
                # Si NO pedimos fondo: [Target] -> (1, H, W)
                final_mask = target_mask[None, :, :]
            
        elif self.classes is not None:
            # === CASO B: Lista de Clases (One-Hot) ===
            masks_list = []
            
            # Generar máscaras para cada clase solicitada
            for cid in self.classes:
                channel = (temp_mask == cid).astype(np.uint8)
                masks_list.append(channel)
            
            if self.add_background:
                # Calcular fondo: Pixeles donde la suma de todas las clases sea 0
                # Stack temporal para sumar
                all_classes_sum = np.sum(np.stack(masks_list, axis=0), axis=0)
                # Clip para asegurar binario (aunque no deberían solaparse)
                all_classes_sum = np.clip(all_classes_sum, 0, 1)
                
                bg_mask = (1 - all_classes_sum).astype(np.uint8)
                
                # Insertar al principio
                masks_list.insert(0, bg_mask)
            
            # Apilar -> (N_Canales, H, W)
            final_mask = np.stack(masks_list, axis=0)
            
        else:
            # === CASO C: Índices Crudos (Sin One-Hot) ===
            # Aquí add_background no aplica igual, porque es un mapa de indices.
            # Convertimos los -1 (fondo vacío) a 0 para que sea compatible con CrossEntropy
            # Pero OJO: esto asume que tu clase 0 es fondo. Si tu clase 0 es Objeto,
            # esto podría corromper datos. Idealmente usar casos A o B.
            temp_mask[temp_mask == -1] = 0 
            final_mask = temp_mask.astype(np.uint8)

        # --- 4. Transponer y Wrappers ---
        image = image.transpose(2, 0, 1) # (3, H, W)
        
        img_tensor = tv_tensors.Image(image)
        # Aseguramos que Mask siempre tenga al menos dimension 3 (C, H, W) si usamos A o B
        if final_mask.ndim == 2:
            mask_tensor = tv_tensors.Mask(final_mask)
        else:
            mask_tensor = tv_tensors.Mask(final_mask)

        # --- 5. Transformaciones ---
        if self.transform:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)
        
        return img_tensor, mask_tensor