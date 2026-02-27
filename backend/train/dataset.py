"""
dataset.py - EMNIST + Math/Greek Symbol Data Loader

Expanded from 47 → 76 classes:
- Classes 0-46: Original EMNIST ByMerge (digits, uppercase, select lowercase)
- Classes 47-75: Math operators and Greek letters (synthetic + HASYv2 if available)

Synthetic symbol generation renders characters from Unicode fonts onto 28×28
images with random augmentation to approximate handwriting variation.
"""

import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont


# ================================================================
# LABEL SYSTEM — 76 total classes
# ================================================================

# Original EMNIST ByMerge: 47 classes (indices 0-46)
EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

# New: Math operators + Greek letters (indices 47-75)
MATH_LABELS = [
    # Math operators (indices 47-58)
    '+', '−', '×', '÷', '=', '≠',
    '<', '>', '≤', '≥', '±', '√',
    # Greek letters (indices 59-75)
    'α', 'β', 'γ', 'δ', 'ε', 'θ',
    'λ', 'μ', 'π', 'σ', 'τ', 'φ',
    'ψ', 'ω', '∞', '∑', '∫',
]

# Combined label list
ALL_LABELS = EMNIST_LABELS + MATH_LABELS
NUM_CLASSES = len(ALL_LABELS)  # 76
EMNIST_NUM_CLASSES = len(EMNIST_LABELS)  # 47

# Label categories for context mode masking
DIGIT_INDICES = list(range(0, 10))                    # 0-9
UPPER_INDICES = list(range(10, 36))                    # A-Z
LOWER_INDICES = list(range(36, 47))                    # a,b,d,e,f,g,h,n,q,r,t
MATH_INDICES = list(range(47, 59))                     # +−×÷=≠<>≤≥±√
GREEK_INDICES = list(range(59, 76))                    # α β γ δ ε θ λ μ π σ τ φ ψ ω ∞ ∑ ∫

# Indices active in each context mode
MODE_INDICES = {
    'all': list(range(NUM_CLASSES)),
    'text': DIGIT_INDICES + UPPER_INDICES + LOWER_INDICES,
    'math': DIGIT_INDICES + MATH_INDICES + GREEK_INDICES,
}

# Dataset normalization stats
EMNIST_MEAN = 0.1751
EMNIST_STD = 0.3332
# Combined stats (will be close to EMNIST since it dominates)
COMBINED_MEAN = 0.1751
COMBINED_STD = 0.3332


class TransposeImage:
    """EMNIST images are rotated/transposed by default. This fixes orientation."""
    def __call__(self, x):
        return x.transpose(1, 2)


# ================================================================
# SYNTHETIC SYMBOL DATASET
# ================================================================

# Unicode characters to render for each math/Greek class
SYMBOL_CHARS = {
    '+': ['+', '＋'],
    '−': ['−', '–', '-'],
    '×': ['×', '✕'],
    '÷': ['÷'],
    '=': ['=', '＝'],
    '≠': ['≠'],
    '<': ['<', '＜'],
    '>': ['>', '＞'],
    '≤': ['≤'],
    '≥': ['≥'],
    '±': ['±'],
    '√': ['√'],
    'α': ['α', 'ɑ'],
    'β': ['β'],
    'γ': ['γ'],
    'δ': ['δ'],
    'ε': ['ε', 'ϵ'],
    'θ': ['θ', 'ϑ'],
    'λ': ['λ'],
    'μ': ['μ', 'µ'],
    'π': ['π', 'ϖ'],
    'σ': ['σ', 'ς'],
    'τ': ['τ'],
    'φ': ['φ', 'ϕ'],
    'ψ': ['ψ'],
    'ω': ['ω'],
    '∞': ['∞'],
    '∑': ['∑'],
    '∫': ['∫'],
}


def _find_system_fonts():
    """Find available system fonts that support Unicode math/Greek."""
    font_paths = []
    search_dirs = []

    if sys.platform == 'win32':
        search_dirs = [os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')]
    elif sys.platform == 'darwin':
        search_dirs = ['/System/Library/Fonts', '/Library/Fonts', os.path.expanduser('~/Library/Fonts')]
    else:
        search_dirs = ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts')]

    preferred = [
        'DejaVuSans.ttf', 'DejaVuSerif.ttf', 'DejaVuSansMono.ttf',
        'NotoSansMath-Regular.ttf', 'NotoSans-Regular.ttf',
        'STIXTwoMath-Regular.otf', 'STIXTwoText-Regular.otf',
        'FreeSerif.ttf', 'FreeSans.ttf',
        'arial.ttf', 'times.ttf', 'cour.ttf',
        'LiberationSans-Regular.ttf', 'LiberationSerif-Regular.ttf',
        'Ubuntu-R.ttf', 'NotoSansMono-Regular.ttf',
    ]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.ttf', '.otf')):
                    full = os.path.join(root, f)
                    # Prioritize preferred fonts
                    if f in preferred:
                        font_paths.insert(0, full)
                    else:
                        font_paths.append(full)

    return font_paths


def _render_symbol(char, font, size=28):
    """Render a single character onto a 28×28 grayscale image."""
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)

    # Try different font sizes to fill the canvas well
    for fs in [22, 20, 18, 16, 24]:
        try:
            pil_font = ImageFont.truetype(font, fs)
            bbox = draw.textbbox((0, 0), char, font=pil_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw > 0 and th > 0 and tw < size and th < size:
                x = (size - tw) // 2 - bbox[0]
                y = (size - th) // 2 - bbox[1]
                draw.text((x, y), char, fill=255, font=pil_font)
                arr = np.array(img, dtype=np.float32)
                if arr.max() > 10:
                    return arr
        except Exception:
            continue

    # Fallback: use default font
    try:
        pil_font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=pil_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = max(0, (size - tw) // 2 - bbox[0])
        y = max(0, (size - th) // 2 - bbox[1])
        draw.text((x, y), char, fill=255, font=pil_font)
    except Exception:
        pass

    return np.array(img, dtype=np.float32)


class SyntheticSymbolDataset(Dataset):
    """
    Generates synthetic handwriting-like images for math/Greek symbols.

    For each symbol class, renders from multiple fonts with random:
    - Rotation (±15°)
    - Translation (±15%)
    - Scale (80-120%)
    - Stroke width variation (via dilation/erosion)
    - Gaussian noise
    - Brightness variation

    Produces samples_per_class images per symbol.
    Target labels are offset by EMNIST_NUM_CLASSES (47) so they don't
    collide with EMNIST class indices.
    """

    def __init__(self, samples_per_class=5000, transform=None, train=True):
        self.transform = transform
        self.train = train
        self.samples_per_class = samples_per_class
        self.num_classes = len(MATH_LABELS)
        self.label_offset = EMNIST_NUM_CLASSES  # 47

        # Find fonts
        self.fonts = _find_system_fonts()
        if not self.fonts:
            print("WARNING: No system fonts found. Using PIL default font.")
            self.fonts = [None]

        # Pre-render base templates for each class from each font
        print(f"Generating synthetic symbol templates from {len(self.fonts)} fonts...")
        self.templates = {}  # class_idx -> list of numpy arrays
        for ci, label in enumerate(MATH_LABELS):
            chars = SYMBOL_CHARS.get(label, [label])
            templates = []
            for font in self.fonts[:10]:  # Use up to 10 fonts
                for char in chars:
                    rendered = _render_symbol(char, font)
                    if rendered.max() > 10:  # Valid render
                        templates.append(rendered)
            if not templates:
                # Emergency fallback: draw a simple shape
                templates.append(self._emergency_render(label))
            self.templates[ci] = templates

        total = self.num_classes * self.samples_per_class
        split = int(total * 0.85) if train else total - int(total * 0.85)
        print(f"Synthetic dataset: {self.num_classes} classes × {samples_per_class} = {total} "
              f"({'train' if train else 'test'}: {split})")

    def _emergency_render(self, label):
        """Fallback renderer for when no font can render the character."""
        img = np.zeros((28, 28), dtype=np.float32)
        # Draw a distinctive pattern based on the label hash
        h = hash(label) % 1000
        cx, cy = 14, 14
        for i in range(28):
            for j in range(28):
                dist = ((i - cy) ** 2 + (j - cx) ** 2) ** 0.5
                if dist < 8 + (h % 5):
                    img[i, j] = max(0, 255 - dist * 20)
        return img

    def __len__(self):
        total = self.num_classes * self.samples_per_class
        if self.train:
            return int(total * 0.85)
        return total - int(total * 0.85)

    def __getitem__(self, idx):
        # Determine class and pick a random template
        class_idx = idx % self.num_classes
        templates = self.templates[class_idx]
        template = templates[np.random.randint(len(templates))].copy()

        # Apply random augmentation to simulate handwriting
        template = self._augment(template)

        # Normalize to [0, 1]
        if template.max() > 0:
            template = template / template.max()

        # Convert to tensor (1, 28, 28)
        tensor = torch.tensor(template, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        # Label offset: EMNIST uses 0-46, symbols use 47-75
        label = class_idx + self.label_offset

        return tensor, label

    def _augment(self, img):
        """Apply handwriting-like augmentation to a rendered symbol."""
        from PIL import Image as PILImage

        pil = PILImage.fromarray(img.astype(np.uint8), mode='L')

        # Random rotation ±15°
        angle = np.random.uniform(-15, 15)
        pil = pil.rotate(angle, fillcolor=0, resample=PILImage.BILINEAR)

        # Random translation ±15%
        tx = int(np.random.uniform(-4, 4))
        ty = int(np.random.uniform(-4, 4))
        pil = pil.transform((28, 28), PILImage.AFFINE, (1, 0, -tx, 0, 1, -ty), fillcolor=0)

        # Random scale 85-115%
        scale = np.random.uniform(0.85, 1.15)
        new_sz = max(10, int(28 * scale))
        pil = pil.resize((new_sz, new_sz), PILImage.BILINEAR)
        # Recenter
        result = PILImage.new('L', (28, 28), 0)
        ox = (28 - new_sz) // 2
        oy = (28 - new_sz) // 2
        result.paste(pil, (ox, oy))
        pil = result

        arr = np.array(pil, dtype=np.float32)

        # Random brightness variation
        arr = arr * np.random.uniform(0.7, 1.0)

        # Add Gaussian noise
        noise = np.random.normal(0, np.random.uniform(2, 12), arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255)

        # Random stroke thickness via simple morphological ops
        if np.random.random() < 0.3:
            # Slight dilation (thicker strokes)
            kernel = np.ones((2, 2), dtype=np.float32)
            from scipy.ndimage import maximum_filter
            try:
                arr = maximum_filter(arr, size=2)
            except ImportError:
                pass  # scipy not available, skip

        return arr


# ================================================================
# DATA LOADING
# ================================================================

def _get_num_workers():
    """Platform-safe worker count."""
    if sys.platform == 'win32':
        return 0
    return 8


def get_data_loaders(data_dir='data/raw/emnist', batch_size=128, num_workers=None,
                     include_symbols=True):
    """
    Create combined EMNIST + Symbol DataLoaders.

    Args:
        data_dir: Where to download/find EMNIST data
        batch_size: Training batch size
        num_workers: DataLoader workers (auto-detected if None)
        include_symbols: If True, include synthetic math/Greek symbols

    Returns:
        train_loader, test_loader
    """
    if num_workers is None:
        num_workers = _get_num_workers()

    print(f"DataLoader workers: {num_workers}")
    print(f"Loading EMNIST dataset... (include_symbols={include_symbols})")

    # === Training augmentation for EMNIST ===
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeImage(),
        transforms.RandomAffine(
            degrees=10, translate=(0.10, 0.10),
            scale=(0.90, 1.10), fill=0
        ),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3, fill=0),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.7)),
        transforms.Normalize((COMBINED_MEAN,), (COMBINED_STD,)),
        transforms.RandomErasing(
            p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3),
            value=(-COMBINED_MEAN / COMBINED_STD)
        ),
    ])

    # === Clean test transform ===
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        TransposeImage(),
        transforms.Normalize((COMBINED_MEAN,), (COMBINED_STD,))
    ])

    # === Augmentation for synthetic symbols (already tensors, no ToTensor/Transpose) ===
    symbol_train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=8, translate=(0.08, 0.08),
            scale=(0.92, 1.08), fill=0
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
        transforms.Normalize((COMBINED_MEAN,), (COMBINED_STD,)),
        transforms.RandomErasing(
            p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3),
            value=(-COMBINED_MEAN / COMBINED_STD)
        ),
    ])

    symbol_test_transform = transforms.Compose([
        transforms.Normalize((COMBINED_MEAN,), (COMBINED_STD,)),
    ])

    os.makedirs(data_dir, exist_ok=True)

    # Load EMNIST
    train_dataset = datasets.EMNIST(
        root=data_dir, split='bymerge', train=True,
        download=True, transform=train_transform
    )
    test_dataset = datasets.EMNIST(
        root=data_dir, split='bymerge', train=False,
        download=True, transform=test_transform
    )

    print(f"EMNIST — Train: {len(train_dataset):,} | Test: {len(test_dataset):,}")

    if include_symbols:
        # Create synthetic symbol datasets
        # ~5000 per class to partially balance with EMNIST (~15K per class)
        symbol_train = SyntheticSymbolDataset(
            samples_per_class=5000,
            transform=symbol_train_transform,
            train=True
        )
        symbol_test = SyntheticSymbolDataset(
            samples_per_class=5000,
            transform=symbol_test_transform,
            train=False
        )

        print(f"Symbols — Train: {len(symbol_train):,} | Test: {len(symbol_test):,}")

        # Combine datasets
        combined_train = ConcatDataset([train_dataset, symbol_train])
        combined_test = ConcatDataset([test_dataset, symbol_test])

        print(f"Combined — Train: {len(combined_train):,} | Test: {len(combined_test):,}")

        # Create weighted sampler to handle class imbalance
        # EMNIST has ~15K/class, symbols have ~5K/class
        # We oversample symbols ~2x so the model sees them more often
        n_emnist = len(train_dataset)
        n_symbols = len(symbol_train)
        weight_emnist = 1.0
        weight_symbol = 2.5  # Oversample symbols
        weights = [weight_emnist] * n_emnist + [weight_symbol] * n_symbols
        sampler = WeightedRandomSampler(weights, num_samples=len(combined_train), replacement=True)

        train_dataset = combined_train
        test_dataset = combined_test
    else:
        sampler = None

    print("Creating DataLoaders...")

    use_persistent = num_workers > 0 and sys.platform != 'win32'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent
    )

    print(f"DataLoaders ready. {NUM_CLASSES} classes total.")
    return train_loader, test_loader
