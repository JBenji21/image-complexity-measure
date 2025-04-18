import streamlit as st
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

def compress_image(img: Image.Image) -> int:
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    return buffer.getbuffer().nbytes

def corrupt_image(img_array: np.ndarray, num_corrupt: int) -> np.ndarray:
    corrupted = img_array.copy()
    h, w, _ = corrupted.shape
    flat = corrupted.reshape(-1, 3)
    indices = np.random.choice(flat.shape[0], size=num_corrupt, replace=False)
    flat[indices] = np.random.randint(0, 256, size=(num_corrupt, 3), dtype=np.uint8)
    return flat.reshape(h, w, 3)

resize_dim = st.slider("Resize image to", min_value=64, max_value=512, value=256, step=32)
def estimate_image_complexity(img: Image.Image, steps=11, trials=5, size=(256, 256)):
    img = img.resize(size, Image.LANCZOS).convert('RGB')
    img_array = np.array(img, dtype=np.uint8)
    h, w, _ = img_array.shape
    n_pixels = h * w

    step_fracs = np.linspace(0.0, 1.0, steps)
    compressed_sizes = []

    for frac in step_fracs:
        keep = int(frac * n_pixels)
        corrupt = n_pixels - keep
        sizes = []
        for _ in range(trials):
            corrupted_array = corrupt_image(img_array, corrupt)
            corrupted_img = Image.fromarray(corrupted_array)
            sizes.append(compress_image(corrupted_img))
        compressed_sizes.append(np.mean(sizes))

    V = np.array(compressed_sizes, dtype=float)
    N = len(V) - 1
    V0, VN = V[0], V[-1]
    A = -N * VN + V[1:].sum()
    B = -V[1:].sum() + N * V0
    C3 = VN * (V0 - VN) * A / (B * 1000000) if B != 0 else 0 # dividing by 1,000,000 so it's in kilobytes^2
    C3_norm = VN * (V0 - VN) * A / (B * V0**2) if V0 != 0 else 0

    return V, V0, VN, A, B, C3, C3_norm

def plot_complexity(V, V0, VN, A, B):
    N = len(V) - 1
    x = np.arange(N + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, V, 'o-', label='V(i)', lw=2)
    ax.hlines(V0, 0, N, colors='gray', linestyles='--', label='V(0)')
    ax.hlines(VN, 0, N, colors='black', linestyles='--', label='V(N)')
    ax.fill_between(x, V, V0, where=V <= V0, color='orange', alpha=0.3, label=f'B = {B:.1f}')
    ax.fill_between(x, V, VN, where=V >= VN, color='skyblue', alpha=0.3, label=f'A = {A:.1f}')
    ax.plot(0, V0, 'ro')
    ax.text(0, V0, ' V(0)', va='bottom', ha='left')
    ax.plot(N, VN, 'ro')
    ax.text(N, VN, ' V(N)', va='bottom', ha='right')
    ax.set_title("Image V(i) with A & B areas")
    ax.set_xlabel("Step i → increasing order (less noise)")
    ax.set_ylabel("V(i) = compressed size (bytes)")
    ax.legend(loc='lower left')
    ax.grid(True)
    fig.tight_layout()
    return fig

# Streamlit app
st.title("Image Complexity Analyzer")

st.markdown("""
This app calculates the **structural complexity** of images using a novel compression-based method.

### 📌 How it works:
- Your image is progressively corrupted with random pixel noise, going from complete randomness (100% noise) to the original image (0% noise).
- At each corruption step, the image is compressed, and the file size (compressed size) is recorded.
- These measurements form a curve that reflects how quickly and non-linearly the structure of your image emerges from randomness.

### 📌 Complexity Metric (C):
The complexity score, denoted **C**, combines three factors:

- **Interdependence (A/B)**: How non-linear the compression curve is (structure emerging synergistically).
- **Final Compression (V(N))**: How small (compressed) the fully structured image is.
- **Compression Gain (V(0) - V(N))**: How much simpler the image becomes as noise is removed.

The resulting metric (**C**) is measured in KiloBytes², reflecting both the depth and complexity of your image's inherent structure.

### 📌 Normalized Complexity:
To compare complexity scores across different images and resolutions fairly, we also provide a **normalized complexity** metric (**Cₙₒᵣₘ**) that divides C by the square of the baseline (fully random) compression size \(V(0)^2\). This normalized measure highlights how structurally complex the image is relative to its size and base entropy.


Upload your image below to explore its complexity, and use the noise-level slider to visualize how the image transforms under increasing randomness.
""")


uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

steps = st.slider("Number of corruption levels", 5, 21, 11)
trials = st.slider("Trials per corruption level", 1, 10, 5)

if uploaded_file:
    # 1) Load and display original
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2) Immediately create a resized version and array for both complexity & noise
    resized_image = image.resize((resize_dim, resize_dim), Image.LANCZOS).convert('RGB')
    img_array = np.array(resized_image, dtype=np.uint8)

    # 3) Compute the complexity curve
    with st.spinner("Analyzing complexity..."):
        V, V0, VN, A, B, C3, C3_norm = estimate_image_complexity(
            image, steps=steps, trials=trials, size=(resize_dim, resize_dim)
        )

    # 4) Show results
    st.markdown(f"**A:** {A:.2f} bytes &nbsp;&nbsp; **B:** {B:.2f} bytes &nbsp;&nbsp; **A/B:** {A/B:.2f}")
    st.markdown(f"""
**Complexity (C)**: `{C3:.2f}` kilobytes²  
**Normalized Complexity (Cₙₒᵣₘ)**: `{C3_norm:.6f}` (unitless, relative)
""")
    st.pyplot(plot_complexity(V, V0, VN, A, B))

    # 5) Noise preview (now safely in scope)
    st.header("Visualize Noise Level")
    noise_level = st.slider("Preview image with noise level (%)", 0, 100, 100, 5)

    total_pixels = resize_dim * resize_dim
    num_corrupt  = int((noise_level / 100) * total_pixels)

    if noise_level < 100:
        corrupted_array   = corrupt_image(img_array, num_corrupt)
        corrupted_preview = Image.fromarray(corrupted_array)
        caption_text      = f"{noise_level}% noise"
    else:
        corrupted_preview = resized_image
        caption_text      = "Original resized image (0% noise)"

    st.image(corrupted_preview, caption=caption_text, use_container_width=True)

else:
    st.info("Upload an image to begin analysis.")

