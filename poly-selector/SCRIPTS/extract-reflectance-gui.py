import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from spectral import envi, get_rgb
from skimage.draw import polygon
import os
from matplotlib.colors import to_rgba
import argparse
import glob
import csv

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
ax_radio = plt.axes([0.02, 0.66, 0.25, 0.25], frameon=True)  # Radio buttons at the top

# Sliders below radio buttons - made shorter to fit labels
ax_low = plt.axes([0.07, 0.53, 0.2, 0.03])
ax_high = plt.axes([0.07, 0.48, 0.2, 0.03])
ax_gain = plt.axes([0.07, 0.43, 0.2, 0.03])
ax_offset = plt.axes([0.07, 0.38, 0.2, 0.03])

# Buttons at the bottom
ax_reset = plt.axes([0.02, 0.25, 0.25, 0.04])
ax_save = plt.axes([0.02, 0.2, 0.25, 0.04])
ax_continue = plt.axes([0.02, 0.15, 0.25, 0.04])
ax_load = plt.axes([0.02, 0.1, 0.25, 0.04])

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

def process_polygon(pts, polygon_num, ax, cube, output_dir, args, used_colors, colors, save_data=True, show_spectrum=True):
    """Process a polygon and extract reflectance data.
    
    Args:
        pts: List of (x,y) coordinates for the polygon vertices
        polygon_num: Number of the polygon being processed
        ax: Matplotlib axis to draw on
        cube: The hyperspectral data cube
        output_dir: Directory to save output files
        args: Command line arguments
        used_colors: List of colors already used for polygons
        colors: List of available colors
        save_data: Whether to save data to CSV files (default: True)
        show_spectrum: Whether to show the spectrum plot (default: True)
    """
    print(f"Processing polygon {polygon_num} with {len(pts)} points")
    r = np.array([p[1] for p in pts])
    c = np.array([p[0] for p in pts])
    rr, cc = polygon(r, c, cube.shape[:2])
    mask = np.zeros(cube.shape[:2], dtype=bool)
    mask[rr, cc] = True

    # Draw the polygon on the image with its assigned color
    color = used_colors[polygon_num - 1] if polygon_num <= len(used_colors) else colors[polygon_num % len(colors)]
    
    # Draw the polygon edges
    ax.plot(c, r, '-', linewidth=2, color=color)
    
    # Draw the closing edge if we have at least 3 points
    if len(pts) > 2:
        ax.plot([c[-1], c[0]], [r[-1], r[0]], '-', linewidth=2, color=color)
    
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

    avg_spectrum = spectra.mean(axis=0)
    std_spectrum = spectra.std(axis=0)

    wavelengths = np.linspace(350, 1000, cube.shape[2])
    
    if show_spectrum:
        # Create a new figure for this polygon's spectrum
        spectrum_fig = plt.figure(figsize=(8, 6))
        spectrum_ax = spectrum_fig.add_subplot(111)
        spectrum_ax.plot(wavelengths, avg_spectrum, label='Mean Reflectance', color=color)
        spectrum_ax.fill_between(wavelengths, avg_spectrum - std_spectrum, avg_spectrum + std_spectrum, 
                               alpha=0.3, color=color, label='Std Dev')
        spectrum_ax.set_title(f"Average Reflectance Spectrum for Polygon {polygon_num}")
        spectrum_ax.set_xlabel("Wavelength (nm)")
        spectrum_ax.set_ylabel("Reflectance")
        spectrum_ax.legend()
        spectrum_ax.grid(True)
        # Position the figure window
        spectrum_fig.canvas.manager.set_window_title(f"Polygon {polygon_num} Spectrum")
        spectrum_fig.canvas.draw()
        plt.show(block=False)

    if save_data:
        # Save normalized polygon coordinates
        # Note: X_coord corresponds to column (shape[1]) and Y_coord to row (shape[0])
        polygon_data = np.column_stack((c / cube.shape[1], r / cube.shape[0]))
        polygon_path = f'{output_dir}/{args.filename}_polygon_{polygon_num}.csv'
        np.savetxt(polygon_path, polygon_data, delimiter=',', header='X_coord,Y_coord', comments='')
        print(f"Saved polygon {polygon_num} coordinates to: {polygon_path}")
        print(f"Polygon data: {polygon_data.shape[0]} rows (vertices), {polygon_data.shape[1]} columns (coordinates)\n")

        # Save spectrum data for the whole polygon (mean + standard deviation)
        output_data = np.column_stack((wavelengths, avg_spectrum, std_spectrum))
        output_path = f'{output_dir}/{args.filename}_spectrum_polygon_{polygon_num}_summary.csv'
        np.savetxt(output_path, output_data, delimiter=',', header='Wavelength (nm),Mean Reflectance,Std Dev', comments='')
        print(f"Saved spectrum summary for polygon {polygon_num} to: {output_path}")
        print(f"Summary spectrum data: {output_data.shape[0]} rows (spectral bands), {output_data.shape[1]} columns (wavelength, mean, st. dev.)\n")

        # Save spectrum data for the subsample (random 100 points)
        # Create header with wavelength and sample numbers
        header = 'Wavelength (nm),' + ','.join([f'Pixel_{i+1}' for i in range(len(subsample))])
        # Stack wavelengths with transposed subsample (each column will be the spectrum of one sample)
        subsample_data = np.column_stack((wavelengths, subsample.T))
        subsample_path = f'{output_dir}/{args.filename}_spectrum_polygon_{polygon_num}_random_sample.csv'
        np.savetxt(subsample_path, subsample_data, delimiter=',', header=header, comments='')
        print(f"Saved spectrum subsample for polygon {polygon_num} to: {subsample_path}")
        print(f"Subsample spectrum data: {subsample_data.shape[0]} rows (spectral bands), {subsample_data.shape[1]} columns (wavelength + 100 points)")

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
            # Draw the closing edge from last point to first point
            if len(current_points) > 2:  # Only draw closing edge if we have at least 3 points
                color = used_colors[-1]  # Use the same color as the polygon
                ax.plot([current_points[-1][0], current_points[0][0]], 
                       [current_points[-1][1], current_points[0][1]], '-', color=color)
                fig.canvas.draw_idle()
            
            print(f"\nProcessing polygon {len(all_pts) + 1} with {len(current_points)} points")
            # Store the current points before clearing
            pts_to_process = current_points.copy()
            all_pts.append(pts_to_process)
            current_points.clear()
            # Process the stored points and save data
            process_polygon(pts_to_process, len(all_pts), ax, cube, output_dir, args, used_colors, colors, save_data=True)
        elif event.key == 'q':
            # Process any remaining points before quitting
            if current_points:
                # Draw the closing edge from last point to first point
                if len(current_points) > 2:  # Only draw closing edge if we have at least 3 points
                    color = used_colors[-1]  # Use the same color as the polygon
                    ax.plot([current_points[-1][0], current_points[0][0]], 
                           [current_points[-1][1], current_points[0][1]], '-', color=color)
                    fig.canvas.draw_idle()
                
                print(f"\nProcessing final polygon {len(all_pts) + 1} with {len(current_points)} points")
                pts_to_process = current_points.copy()
                all_pts.append(pts_to_process)
                process_polygon(pts_to_process, len(all_pts), ax, cube, output_dir, args, used_colors, colors, save_data=True)
            print(f"\nTotal polygons processed: {len(all_pts)}")
            drawing_polygon = False
            plt.disconnect(cid_click)
            plt.disconnect(cid_key)

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    print("Draw your polygons. Click to add points, press Enter to finish each polygon, 'q' to quit.")

def load_polygons(event):
    # Load existing polygon coordinates from CSV files and draw them on the image
    global all_pts, drawing_polygon, final_rgb
    
    # Process the RGB image first
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val
    p_low, p_high = np.percentile(rgb_raw, (low, high))
    final_rgb = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    final_rgb = np.clip(gain * final_rgb + offset, 0, 1)
    
    # Check for existing polygon CSV files in the output directory
    pattern = f'{output_dir}/{args.filename}_polygon_*.csv'
    polygon_files = glob.glob(pattern)
    
    if polygon_files:
        print(f"\nFound {len(polygon_files)} existing polygon file(s).")
        print("Files found:", polygon_files)
        
        # Clear the current image and show the final RGB
        ax.clear()
        ax.imshow(final_rgb)
        ax.set_title("Loaded polygons from CSV files")
        
        # Create a colormap with 10 distinct colors from hsv
        colors = plt.cm.hsv(np.linspace(0, 1, 10))
        # Create a list to track used colors
        used_colors = []
        # Shuffle the colors for random selection
        np.random.shuffle(colors)
        
        all_pts = []
        # Sort files by polygon number to ensure consistent color assignment
        polygon_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print("Sorted files:", polygon_files)
        
        # First pass: draw all polygons without showing spectra
        for file in polygon_files:
            # Extract polygon number from filename
            polygon_num = int(file.split('_')[-1].split('.')[0])
            print(f"\nProcessing polygon {polygon_num} from file: {file}")
            
            # Read the CSV file
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                coords = [(float(row['X_coord']), float(row['Y_coord'])) for row in reader]
            print(f"Read {len(coords)} coordinates from file")
            
            # Convert normalized coordinates back to image coordinates
            # Note: X_coord corresponds to column (shape[1]) and Y_coord to row (shape[0])
            pts = [(c[0] * cube.shape[1], c[1] * cube.shape[0]) for c in coords]
            all_pts.append(pts)
            
            # Add color to used_colors if not already present
            if polygon_num > len(used_colors):
                used_colors.append(colors[(polygon_num - 1) % len(colors)])
            
            # Process the polygon without saving data and without showing spectrum
            process_polygon(pts, polygon_num, ax, cube, output_dir, args, used_colors, colors, save_data=False, show_spectrum=False)
            print(f"Finished processing polygon {polygon_num}")
        
        fig.canvas.draw_idle()
        print(f"\nLoaded {len(all_pts)} polygons from CSV files.")
        
        # Create a new figure for the button dialog
        dialog_fig = plt.figure(figsize=(4, 2))
        dialog_ax = dialog_fig.add_subplot(111)
        dialog_ax.text(0.5, 0.6, "Would you like to see the spectrum plots?",
                      ha='center', va='center', transform=dialog_ax.transAxes)
        dialog_ax.set_axis_off()
        
        # Create buttons
        ax_yes = plt.axes([0.3, 0.2, 0.2, 0.2])
        ax_no = plt.axes([0.6, 0.2, 0.2, 0.2])
        yes_button = Button(ax_yes, 'Yes')
        no_button = Button(ax_no, 'No')
        
        def on_yes(event):
            plt.close(dialog_fig)
            # Enable interactive mode for non-blocking plots
            plt.ion()
            # Second pass: show spectra for each polygon
            for i, pts in enumerate(all_pts, 1):
                process_polygon(pts, i, ax, cube, output_dir, args, used_colors, colors, save_data=False, show_spectrum=True)
            # Keep the plots open
            plt.ioff()
            plt.show(block=False)
        
        def on_no(event):
            plt.close(dialog_fig)
        
        yes_button.on_clicked(on_yes)
        no_button.on_clicked(on_no)
        plt.show(block=True)
    else:
        print("\nNo existing polygon files found.")

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
load_button = Button(ax_load, 'Load Polygon Data')
load_button.on_clicked(load_polygons)
radio.on_clicked(change_band)

plt.show()
