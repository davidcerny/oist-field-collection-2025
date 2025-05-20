import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from spectral import envi, get_rgb
from skimage.draw import polygon
import os
from matplotlib.colors import to_rgba
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract reflectance data from ENVI image using polygon selection.')
parser.add_argument('-f', '--filename', required=True, help='Base filename of the ENVI image (without extension)')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

# Check for and create OUTPUT directory if it doesn't exist
output_dir = os.path.join(script_dir, '..', 'OUTPUT')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created OUTPUT directory at: {output_dir}")

# Construct paths using the provided filename
hdr_path = os.path.join(script_dir, '..', 'DATA', args.filename + '.hdr')
bin_path = os.path.join(script_dir, '..', 'DATA', args.filename + '.bin')

# Load ENVI image and binary data
data = envi.open(hdr_path, image=bin_path)
cube = data.load()

# Print image resolution
print(f"\nImage resolution: {cube.shape[0]} x {cube.shape[1]} pixels")
print(f"Number of spectral bands: {cube.shape[2]}\n")

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
plt.subplots_adjust(left=0.15, bottom=0.1, right=1.15)
img_disp = ax.imshow(rgb_raw, vmin=0, vmax=1)
ax.set_title("Adjust sliders, then click 'Continue to Polygon'")

# GUI elements - all on the left side
ax_radio = plt.axes([0.02, 0.65, 0.25, 0.25], frameon=True)  # Radio buttons at the top

# Sliders below radio buttons - made shorter to fit labels
ax_low = plt.axes([0.07, 0.5, 0.2, 0.03])
ax_high = plt.axes([0.07, 0.45, 0.2, 0.03])
ax_gain = plt.axes([0.07, 0.4, 0.2, 0.03])
ax_offset = plt.axes([0.07, 0.35, 0.2, 0.03])

# Buttons at the bottom
ax_reset = plt.axes([0.02, 0.2, 0.25, 0.04])
ax_save = plt.axes([0.02, 0.15, 0.25, 0.04])
ax_continue = plt.axes([0.02, 0.1, 0.25, 0.04])

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
    out_path = '{output_dir}/{args.filename}current_rgb_preview.png'
    plt.imsave(out_path, img_disp.get_array())
    print(f"Saved RGB preview to: {out_path}")

def change_band(label):
    global rgb_raw
    selected = band_options[label]
    rgb_raw = get_rgb(data, selected).astype(np.float32)
    update()

def continue_to_polygon(event):
    global final_rgb, current_points, drawing_polygon
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val
    p_low, p_high = np.percentile(rgb_raw, (low, high))
    final_rgb = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    final_rgb = np.clip(gain * final_rgb + offset, 0, 1)

    ax.clear()
    ax.imshow(final_rgb)
    ax.set_title("Draw polygons (Click to add points, press Enter to finish polygon, 'q' to quit)")
    fig.canvas.draw_idle()

    all_pts = []
    current_points = []
    drawing_polygon = True

    # Create a colormap with 10 distinct colors from hsv
    colors = plt.cm.hsv(np.linspace(0, 1, 10))
    # Create a list to track used colors
    used_colors = []
    # Shuffle the colors for random selection
    np.random.shuffle(colors)

    def get_next_color():
        if len(used_colors) < len(colors):
            # If we haven't used all colors, pick a new one
            color = colors[len(used_colors)]
            used_colors.append(color)
            return color
        else:
            # If we've used all colors, start recycling
            return colors[len(used_colors) % len(colors)]

    def on_click(event):
        if event.inaxes != ax or not drawing_polygon:
            return
        current_points.append((event.xdata, event.ydata))
        # Use the color for the current polygon number
        color = get_next_color() if len(current_points) == 1 else used_colors[-1]
        ax.plot(event.xdata, event.ydata, 'o', color=color)
        if len(current_points) > 1:
            ax.plot([current_points[-2][0], current_points[-1][0]], 
                   [current_points[-2][1], current_points[-1][1]], '-', color=color)
        fig.canvas.draw_idle()

    def on_key(event):
        global drawing_polygon, current_points
        if event.key == 'enter' and current_points:
            print(f"\nProcessing polygon {len(all_pts) + 1} with {len(current_points)} points")
            # Store the current points before clearing
            pts_to_process = current_points.copy()
            all_pts.append(pts_to_process)
            current_points.clear()
            # Process the stored points
            process_polygon(pts_to_process, len(all_pts))
        elif event.key == 'q':
            # Process any remaining points before quitting
            if current_points:
                print(f"\nProcessing final polygon {len(all_pts) + 1} with {len(current_points)} points")
                pts_to_process = current_points.copy()
                all_pts.append(pts_to_process)
                process_polygon(pts_to_process, len(all_pts))
            print(f"\nTotal polygons processed: {len(all_pts)}")
            drawing_polygon = False
            plt.disconnect(cid_click)
            plt.disconnect(cid_key)

    def process_polygon(pts, polygon_num):
        print(f"Processing polygon {polygon_num} with {len(pts)} points")
        r = np.array([p[1] for p in pts])
        c = np.array([p[0] for p in pts])
        rr, cc = polygon(r, c, cube.shape[:2])
        mask = np.zeros(cube.shape[:2], dtype=bool)
        mask[rr, cc] = True

        # Draw the polygon on the image with its assigned color
        color = used_colors[polygon_num - 1] if polygon_num <= len(used_colors) else colors[polygon_num % len(colors)]
        ax.plot(c, r, '-', linewidth=2, color=color)
        
        # Calculate and plot the centroid with the label
        centroid_x = np.mean(c)
        centroid_y = np.mean(r)
        ax.text(centroid_x, centroid_y, str(polygon_num), 
                color=color, fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'))
        
        fig.canvas.draw_idle()

        # Get all points in the polygon
        spectra = cube[mask, :]
        
        # Randomly sample 100 points (or all points if less than 100)
        n_samples = min(100, len(spectra))
        if n_samples < len(spectra):
            # Get random indices without replacement
            sample_indices = np.random.choice(len(spectra), n_samples, replace=False)
            subsample = spectra[sample_indices]
        else:
            subsample = spectra

        print(f"Subsample size: {len(subsample)}")

        avg_spectrum = spectra.mean(axis=0)
        std_spectrum = spectra.std(axis=0)

        wavelengths = np.linspace(350, 1000, cube.shape[2])
        plt.figure()
        plt.plot(wavelengths, avg_spectrum, label='Mean Reflectance')
        plt.fill_between(wavelengths, avg_spectrum - std_spectrum, avg_spectrum + std_spectrum, alpha=0.3, label='Std Dev')
        plt.title(f"Average Reflectance Spectrum with Standard Deviation for Polygon {polygon_num}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save normalized polygon coordinates
        polygon_data = np.column_stack((r / cube.shape[0], c / cube.shape[1]))
        polygon_path = f'{output_dir}/{args.filename}polygon_{polygon_num}.csv'
        np.savetxt(polygon_path, polygon_data, delimiter=',', header='X_coord,Y_coord', comments='')
        print(f"Saved polygon {polygon_num} coordinates to: {polygon_path}")
        print(f"Polygon data: {polygon_data.shape[0]} rows (vertices), {polygon_data.shape[1]} columns (coordinates)\n")

        # Save spectrum data for the whole polygon (mean + standard deviation)
        output_data = np.column_stack((wavelengths, avg_spectrum, std_spectrum))
        output_path = f'{output_dir}/{args.filename}spectrum_polygon_{polygon_num}.csv'
        np.savetxt(output_path, output_data, delimiter=',', header='Wavelength (nm),Mean Reflectance,Std Dev', comments='')
        print(f"Saved spectrum summary for polygon {polygon_num} to: {output_path}")
        print(f"Summary spectrum data: {output_data.shape[0]} rows (spectral bands), {output_data.shape[1]} columns (wavelength, mean, st. dev.)\n")

        # Save spectrum data for the subsample (random 100 points)
        # Create header with wavelength and sample numbers
        header = 'Wavelength (nm),' + ','.join([f'Pixel_{i+1}' for i in range(len(subsample))])
        # Stack wavelengths with transposed subsample (each column will be the spectrum of one sample)
        subsample_data = np.column_stack((wavelengths, subsample.T))
        subsample_path = f'{output_dir}/{args.filename}spectrum_polygon_{polygon_num}_random_sample.csv'
        np.savetxt(subsample_path, subsample_data, delimiter=',', header=header, comments='')
        print(f"Saved spectrum subsample for polygon {polygon_num} to: {subsample_path}")
        print(f"Subsample spectrum data: {subsample_data.shape[0]} rows (spectral bands), {subsample_data.shape[1]} columns (wavelength + 100 points)\n")

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    print("Draw your polygons. Click to add points, press Enter to finish each polygon, 'q' to quit.")

# Connect widgets
low_slider.on_changed(update)
high_slider.on_changed(update)
gain_slider.on_changed(update)
offset_slider.on_changed(update)
reset_button = Button(ax_reset, 'Reset')
reset_button.on_clicked(reset)
save_button = Button(ax_save, 'Save RGB')
save_button.on_clicked(save_rgb)
continue_button = Button(ax_continue, 'Continue to Polygon')
continue_button.on_clicked(continue_to_polygon)
radio.on_clicked(change_band)

plt.show()
