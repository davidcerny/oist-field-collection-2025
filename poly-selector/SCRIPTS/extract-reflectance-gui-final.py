
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from spectral import envi, get_rgb
from skimage.draw import polygon
import os

# File paths
hdr_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/test.hdr'. # make paths relative and put data files in DATA folder
bin_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/test.bin'

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
radio = RadioButtons(ax_radio, band_options.keys())

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
    ax.set_title("Draw a polygon around the patch (Right-click or Enter to finish)")
    fig.canvas.draw_idle()

    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

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
    output_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/average_spectrum_with_std.csv'
    np.savetxt(output_path, output_data, delimiter=',', header='Wavelength (nm),Mean Reflectance,Std Dev', comments='')
    print(f"Spectrum with standard deviation saved to: {output_path}")

# Connect widgets
low_slider.on_changed(update)
high_slider.on_changed(update)
gain_slider.on_changed(update)
offset_slider.on_changed(update)
Button(ax_reset, 'Reset').on_clicked(reset)
Button(ax_save, 'Save RGB').on_clicked(save_rgb)
Button(ax_continue, 'Continue to Polygon').on_clicked(continue_to_polygon)
radio.on_clicked(change_band)

plt.show()
