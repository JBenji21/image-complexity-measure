import streamlit as st
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

# --- Streamlit App Header ---
st.title("Image Complexity Analyzer")

st.markdown("""
This app calculates the **structural complexity** of images using a novel compression-based method.

### 📌 How it works:
- Your image is progressively corrupted with random pixel noise, or scrambled, going from complete randomness (100% corruption) to the original image (0% corruption).
- At each step, the image is compressed (PNG, JPEG), and the file size recorded.
- The resulting curve reflects how structure emerges from disorder.

### 📌 Complexity Metric (C):
**C** = (A/B) × V(N) × ((V(0) - V(N)) / V(0))

- **V(0)**: Baseline size (fully corrupted)  
- **V(N)**: Structured size (original)  
- **A/B**: Emergence factor (nonlinearity)  
- **(V(0)-V(N))/V(0)**: Structure spread

Normalized complexity **C_norm** = C / V(0)
""")

# --- Options ---
# … your header and markdown …

# --- Parameter Sliders ---
resize_dim = st.slider("Resize image to", 64, 512, 256, 32)
steps = st.slider("Number of corruption levels", 5, 21, 11)
trials = st.slider("Trials per corruption level", 1, 10, 5)

# === ADVANCED OPTIONS ===
with st.expander("Advanced options"):
    # Compression format (PNG vs JPEG)
    compression_format = st.selectbox(
        "Compression format", 
        ["PNG (default - lossless)", "JPEG (experimental - lossy)"]
    )
    format_str = "JPEG" if "JPEG" in compression_format else "PNG"
    quality = 85
    if format_str == "JPEG":
        quality = st.slider("JPEG quality", 30, 100, 85)

    # Corruption type (noise vs scramble)
    corruption_type = st.radio(
        "Corruption type",
        ["Random noise", "Scramble pixel positions - experimental"]
    )

# === END ADVANCED OPTIONS ===

# --- Core Functions ---
def compress_image(img: Image.Image, fmt: str = 'PNG', quality: int = 85) -> int:
    buffer = io.BytesIO()
    if fmt == 'JPEG':
        img.save(buffer, format='JPEG', optimize=True, quality=quality)
    else:
        img.save(buffer, format='PNG', optimize=True)
    return buffer.getbuffer().nbytes


def corrupt_image(img_array: np.ndarray, num_corrupt: int) -> np.ndarray:
    corrupted = img_array.copy()
    flat = corrupted.reshape(-1, 3)
    idx = np.random.choice(flat.shape[0], size=num_corrupt, replace=False)
    flat[idx] = np.random.randint(0, 256, size=(num_corrupt, 3), dtype=np.uint8)
    return flat.reshape(corrupted.shape)


def scramble_image(img_array: np.ndarray, num_scramble: int) -> np.ndarray:
    scrambled = img_array.copy()
    flat = scrambled.reshape(-1, 3)
    idx = np.random.choice(flat.shape[0], size=num_scramble, replace=False)
    pixels = flat[idx].copy()
    np.random.shuffle(pixels)
    flat[idx] = pixels
    return flat.reshape(scrambled.shape)


def estimate_image_complexity(img: Image.Image,
                              steps=11, trials=5,
                              size=(256, 256),
                              fmt='PNG', quality=85,
                              corruption_type='Random noise'):
    img = img.resize(size, Image.LANCZOS).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    n_pixels = arr.shape[0] * arr.shape[1]

    step_fracs = np.linspace(0.0, 1.0, steps)
    comp_sizes = []
    for frac in step_fracs:
        keep = int(frac * n_pixels)
        corrupt = n_pixels - keep
        sizes = []
        for _ in range(trials):
            if corruption_type == "Scramble pixel positions":
                mod = scramble_image(arr, corrupt)
            else:
                mod = corrupt_image(arr, corrupt)
            img_mod = Image.fromarray(mod)
            sizes.append(compress_image(img_mod, fmt, quality))
        comp_sizes.append(np.mean(sizes))

    V = np.array(comp_sizes, dtype=float)
    N = len(V) - 1
    V0, VN = V[0], V[-1]
    A = -(N+1) * VN + V.sum()
    B = -V.sum() + (N+1) * V0
    EF = A / B if B != 0 else 0
    AC = VN
    SS = (V0 - VN) / V0 if V0 != 0 else 0
    C = EF * AC * SS
    C_norm = C / V0 if V0 != 0 else 0
    return V, V0, VN, A, B, EF, AC, SS, C, C_norm


def plot_complexity(V, V0, VN, A, B):
    x = np.arange(len(V))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, V, 'o-', lw=2, label='V(i)')
    ax.hlines(V0, 0, x[-1], colors='gray', ls='--', label='V(0)')
    ax.hlines(VN, 0, x[-1], colors='black', ls='--', label='V(N)')
    ax.fill_between(x, V, V0, where=V<=V0, color='orange', alpha=0.3, label=f'B')
    ax.fill_between(x, V, VN, where=V>=VN, color='skyblue', alpha=0.3, label=f'A')
    ax.set_xlabel('Step i (less noise)')
    ax.set_ylabel('Compressed size (bytes)')
    ax.legend(loc='lower left')
    ax.grid(True)
    fig.tight_layout()
    return fig

# --- File Uploader & Execution ---
up = st.file_uploader("Choose an image file", type=["png","jpg","jpeg"])
if up:
    img = Image.open(up).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)
    with st.spinner("Analyzing..."):
        V, V0, VN, A, B, EF, AC, SS, C, C_norm = estimate_image_complexity(
            img, steps, trials,
            size=(resize_dim, resize_dim),
            fmt=format_str, quality=quality,
            corruption_type=corruption_type
        )
    st.markdown(f"""
**Emergent Structural Complexity (C):** `{C:.2f}` bytes  
**Normalized Complexity (Cₙₒᵣₘ):** `{C_norm:.6f}`
---
**Baseline size (V₀):** `{V0:.1f}` bytes  
**Structured size (Vₙ):** `{VN:.1f}` bytes  
**Emergence Factor (A/B):** `{EF:.2f}`  
**Absolute Complexity (Vₙ):** `{AC:.1f}` bytes  
**Structure Spread:** `{SS:.4f}`  
""")
    st.pyplot(plot_complexity(V, V0, VN, A, B))
    st.header("Visualize Corruption Level")
    lvl = st.slider("Corruption level (%)", 0, 100, 100, 5)
    corrupt = int((lvl/100) * resize_dim * resize_dim)
    if lvl>0:
        if corruption_type == "Scramble pixel positions":
            arr_mod = scramble_image(np.array(img.resize((resize_dim,resize_dim)).convert('RGB')), corrupt)
        else:
            arr_mod = corrupt_image(np.array(img.resize((resize_dim,resize_dim)).convert('RGB')), corrupt)
        if format_str=="JPEG":
            buf = io.BytesIO()
            Image.fromarray(arr_mod).save(buf, format='JPEG', quality=quality, optimize=True)
            buf.seek(0)
            prev = Image.open(buf)
        else:
            prev = Image.fromarray(arr_mod)
        st.image(prev, caption=f"{lvl}% corrupted", use_container_width=True)
    else:
        st.image(img.resize((resize_dim, resize_dim)), caption="Original (0% corruption)", use_container_width=True)
else:
    st.info("Upload an image to begin analysis.")
