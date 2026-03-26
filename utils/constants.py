"""Dataset constants for 3D point cloud classification."""

# ModelNet40 class names in canonical order
MODELNET40_CLASSES = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]


def get_class_names(num_classes):
    """Get class names for dataset.

    Args:
        num_classes: Number of classes (40 for ModelNet40, etc.)

    Returns:
        List of class names
    """
    if num_classes == 40:
        return MODELNET40_CLASSES
    else:
        return [f"class_{i}" for i in range(num_classes)]
