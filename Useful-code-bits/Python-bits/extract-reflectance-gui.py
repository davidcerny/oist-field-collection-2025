
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from spectral import envi, get_rgb
from skimage.draw import polygon
import os

# File paths
hdr_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/test.hdr'
bin_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/test.bin'

# Load ENVI image and binary data
data = envi.open(hdr_path, image=bin_path)
cube = data.load()

# Get raw RGB and setup interactive adjustment
rgb_raw = get_rgb(data, [20, 40, 60]).astype(np.float32)

# Display figure with sliders
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.25)
img_disp = ax.imshow(rgb_raw)
ax.set_title("Adjust sliders then draw a polygon (Right-click or Enter to finish)")

# Slider axes
ax_low = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_high = plt.axes([0.15, 0.06, 0.65, 0.03])
ax_gain = plt.axes([0.15, 0.02, 0.3, 0.03])
ax_offset = plt.axes([0.55, 0.02, 0.3, 0.03])

# Sliders
low_slider = Slider(ax_low, 'Low %', 0, 10, valinit=1)
high_slider = Slider(ax_high, 'High %', 90, 100, valinit=99)
gain_slider = Slider(ax_gain, 'Gain', 0.5, 2.0, valinit=1)
offset_slider = Slider(ax_offset, 'Offset', -0.5, 0.5, valinit=0)

def update(val):
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val

    p_low, p_high = np.percentile(rgb_raw, (low, high))
    rgb_adj = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    rgb_adj = np.clip(gain * rgb_adj + offset, 0, 1)
    img_disp.set_data(rgb_adj)
    fig.canvas.draw_idle()

# Link sliders to update function
low_slider.on_changed(update)
high_slider.on_changed(update)
gain_slider.on_changed(update)
offset_slider.on_changed(update)

plt.show()

# After window closes, use current slider values to finalize adjusted RGB
low = low_slider.val
high = high_slider.val
gain = gain_slider.val
offset = offset_slider.val

p_low, p_high = np.percentile(rgb_raw, (low, high))
rgb = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
rgb = np.clip(gain * rgb + offset, 0, 1)

# Re-display final image for polygon selection
plt.figure(figsize=(12, 8))
plt.imshow(rgb)
plt.title("Draw a polygon around the color patch (Right-click or Enter to finish)")
pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
plt.close()

# Create mask from polygon
r = np.array([p[1] for p in pts])
c = np.array([p[0] for p in pts])
rr, cc = polygon(r, c, cube.shape[:2])
mask = np.zeros(cube.shape[:2], dtype=bool)
mask[rr, cc] = True

# Extract spectra from ROI
spectra = cube[mask, :]
avg_spectrum = spectra.mean(axis=0)
std_spectrum = spectra.std(axis=0)

# Generate wavelength array and plot
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

# Save results
output_data = np.column_stack((wavelengths, avg_spectrum, std_spectrum))
output_path = '/Users/rosamariorduna/Downloads/binandhdr_folder/average_spectrum_with_std.csv'
np.savetxt(output_path, output_data, delimiter=',', header='Wavelength (nm),Mean Reflectance,Std Dev', comments='')
print(f"Spectrum with standard deviation saved to: {output_path}")
