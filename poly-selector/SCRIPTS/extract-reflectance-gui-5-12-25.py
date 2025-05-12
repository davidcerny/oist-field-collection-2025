
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from spectral import envi, get_rgb
from skimage.draw import polygon
import os


script_dir = os.path.dirname(os.path.abspath(__file__))


hdr_path = os.path.join(script_dir, '..', 'DATA', 'RO_004_5_2025-04-17_01-08-18_.hdr')
bin_path = os.path.join(script_dir, '..', 'DATA', 'RO_004_5_2025-04-17_01-08-18_.bin')


# Load ENVI image and binary data
data = envi.open(hdr_path, image=bin_path)
cube = data.load()

# Define RGB combinations
band_options = {
    'Default (20,40,60)': [20, 40, 60],
    'Alt 1 (10,30,50)': [10, 30, 50],
    'Alt 2 (30,50,70)': [30, 50, 70]
}
selected_bands = band_options['Default (20,40,60)']
rgb_raw = get_rgb(data, selected_bands).astype(np.float32)

# Setup figure and sliders
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.3, bottom=0.4)
img_disp = ax.imshow(rgb_raw, vmin=0, vmax=1)
ax.set_title("Adjust sliders, then click 'Continue to Polygon'")

# GUI elements
ax_low = plt.axes([0.35, 0.26, 0.55, 0.03])
ax_high = plt.axes([0.35, 0.22, 0.55, 0.03])
ax_gain = plt.axes([0.35, 0.18, 0.55, 0.03])
ax_offset = plt.axes([0.35, 0.14, 0.55, 0.03])
ax_reset = plt.axes([0.82, 0.02, 0.12, 0.04])
ax_save = plt.axes([0.65, 0.02, 0.15, 0.04])
ax_continue = plt.axes([0.35, 0.02, 0.25, 0.04])
ax_radio = plt.axes([0.02, 0.6, 0.25, 0.25], frameon=True)

# Sliders
low_slider = Slider(ax_low, 'Low %', 0, 10, valinit=1)
high_slider = Slider(ax_high, 'High %', 90, 100, valinit=99)
gain_slider = Slider(ax_gain, 'Gain', 0.5, 2.0, valinit=1)
offset_slider = Slider(ax_offset, 'Offset', -0.5, 0.5, valinit=0)
radio = RadioButtons(ax_radio, list(band_options.keys()))

# Global for final image
final_rgb = None

def update(val=None):
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val
    p_low, p_high = np.percentile(rgb_raw, (low, high))
    rgb_adj = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    rgb_adj = np.clip(gain * rgb_adj + offset, 0, 1)
    img_disp.set_data(rgb_adj)
    fig.canvas.draw_idle()

def reset(event):
    low_slider.reset()
    high_slider.reset()
    gain_slider.reset()
    offset_slider.reset()

def save_rgb(event):
    update()
    out_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/current_rgb_preview.png'
    plt.imsave(out_path, img_disp.get_array())
    print(f"Saved RGB preview to: {out_path}")

def change_band(label):
    global rgb_raw
    selected = band_options[label]
    rgb_raw = get_rgb(data, selected).astype(np.float32)
    update()

def continue_to_polygon(event):
    global final_rgb
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val
    p_low, p_high = np.percentile(rgb_raw, (low, high))
    final_rgb = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    final_rgb = np.clip(gain * final_rgb + offset, 0, 1)



    ax.clear()
    ax.imshow(final_rgb)
    ax.set_title("Draw polygons (Right-click or press Enter to finish each). Press 'q' in terminal to quit.")
    fig.canvas.draw_idle()

    all_pts = []
    print("Draw your polygons. Press 'q' in the terminal to stop.")

    while True:
        pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
        if not pts:
            break  # no clicks = quit

        all_pts.append(pts)
        r = np.array([p[1] for p in pts])
        c = np.array([p[0] for p in pts])
        rr, cc = polygon(r, c, cube.shape[:2])
        mask = np.zeros(cube.shape[:2], dtype=bool)
        mask[rr, cc] = True

        spectra = cube[mask, :]
        avg_spectrum = spectra.mean(axis=0)
        std_spectrum = spectra.std(axis=0)

        wavelengths = np.linspace(350, 1000, cube.shape[2])
        plt.figure()
        plt.plot(wavelengths, avg_spectrum, label='Mean Reflectance')
        plt.fill_between(wavelengths, avg_spectrum - std_spectrum, avg_spectrum + std_spectrum, alpha=0.3, label='Std Dev')
        plt.title("Average Reflectance Spectrum with Standard Deviation")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.grid(True)
        plt.show()

        output_data = np.column_stack((wavelengths, avg_spectrum, std_spectrum))
        output_path = f'/Users/rosamariorduna/Downloads/binandhdr_folder/spectrum_polygon_{len(all_pts)}.csv'
        np.savetxt(output_path, output_data, delimiter=',', header='Wavelength (nm),Mean Reflectance,Std Dev', comments='')
        print(f"Spectrum saved to: {output_path}")
