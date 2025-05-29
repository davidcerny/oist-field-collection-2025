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

# Define wavelength ranges for RGB (in nanometers)
wavelengths = np.linspace(350, 1000, cube.shape[2])
red_range = (620, 750)    # Red light
green_range = (495, 570)  # Green light
blue_range = (450, 495)   # Blue light

# Define RGB combinations for single-band visualization
band_options = {
    'Default (54,32,22)': [54, 32, 22],
    'Reverse (20,40,60)': [20, 40, 60]
}

def get_band_indices(wavelengths, range_min, range_max):
    # Get indices of bands within a wavelength range
    return np.where((wavelengths >= range_min) & (wavelengths <= range_max))[0]

def resample_band(cube, band_indices):
    # Resample a range of bands by averaging
    return np.mean(cube[:, :, band_indices], axis=2)

def create_resampled_rgb():
    # Get band indices for RGB
    red_indices = get_band_indices(wavelengths, *red_range)
    green_indices = get_band_indices(wavelengths, *green_range)
    blue_indices = get_band_indices(wavelengths, *blue_range)

    # Resample bands for RGB
    red_band = resample_band(cube, red_indices)
    green_band = resample_band(cube, green_indices)
    blue_band = resample_band(cube, blue_indices)

    # Stack the bands to create RGB image
    rgb = np.stack([red_band, green_band, blue_band], axis=2).astype(np.float32)

    # Normalize each band independently
    for i in range(3):
        band = rgb[:, :, i]
        min_val = np.percentile(band, 1)  # Use 1st percentile to avoid outliers
        max_val = np.percentile(band, 99)  # Use 99th percentile to avoid outliers
        rgb[:, :, i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    
    return rgb

def create_single_band_rgb(selected_bands):
    # Create RGB using specific bands
    rgb = get_rgb(data, selected_bands).astype(np.float32)
    return rgb

# Initialize with resampled RGB
rgb_raw = create_resampled_rgb()
my_wavelengths = wavelengths  # Keep the original wavelengths for spectrum plotting

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

# Add radio buttons for visualization method
vis_method = RadioButtons(ax_radio, ['Band Resampling', 'Single Bands (54, 32, 22)', 'Single Bands (20, 40, 60)'])

# Global for final image
final_rgb = None

# Add these global variables at the top of the file, after the imports
all_polygons = {}  # Dictionary to store all polygons and their data
spectrum_figs = {}  # Dictionary to store spectrum plot windows

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
    out_path = f'{output_dir}/{args.filename}_current_rgb_preview.png'
    plt.imsave(out_path, img_disp.get_array())
    print(f"Saved RGB preview to: {out_path}")

def change_vis_method(label):
    global rgb_raw
    if label == 'Band Resampling':
        rgb_raw = create_resampled_rgb()
    elif label == 'Single Bands (54, 32, 22)':
        rgb_raw = create_single_band_rgb(band_options['Default (54,32,22)'])
    else:
        rgb_raw = create_single_band_rgb(band_options['Reverse (20,40,60)'])
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
    
    # Get coordinates of all points in the polygon
    # Create arrays of x and y coordinates for all pixels in the polygon
    y_coords, x_coords = np.where(mask)
    coords = np.column_stack((x_coords, y_coords))  # x, y coordinates for each pixel
    
    # Randomly sample 100 points (or all points if less than 100)
    n_samples = min(100, len(spectra))
    if n_samples < len(spectra):
        # Get random indices without replacement
        sample_indices = np.random.choice(len(spectra), n_samples, replace=False)
        subsample = spectra[sample_indices]
        subsample_coords = coords[sample_indices]  # Get coordinates for sampled points
    else:
        subsample = spectra
        subsample_coords = coords

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

        # Save coordinates and spectrum data for the subsample (random 100 points)
        # Create header with coordinates and wavelengths
        header = 'X_coord,Y_coord,' + ','.join([f'{w:.1f}' for w in wavelengths])
        # Stack coordinates with spectral data
        subsample_data = np.column_stack((subsample_coords, subsample))
        subsample_path = f'{output_dir}/{args.filename}_spectrum_polygon_{polygon_num}_random_sample.csv'
        # Create format string: integers for coordinates, scientific notation for spectra
        fmt = ['%d', '%d'] + ['%.6e'] * subsample.shape[1]
        np.savetxt(subsample_path, subsample_data, delimiter=',', header=header, comments='', fmt=fmt)
        print(f"Saved spectrum subsample for polygon {polygon_num} to: {subsample_path}")
        print(f"Subsample spectrum data: {subsample_data.shape[0]} rows (pixels), {subsample_data.shape[1]} columns (coordinates + wavelengths)")

def get_next_polygon_number(output_dir, args):
    """Get the next polygon number by finding the highest existing number and adding 1."""
    pattern = f'{output_dir}/{args.filename}_polygon_*.csv'
    polygon_files = glob.glob(pattern)
    if not polygon_files:
        return 1
    
    # Extract numbers from filenames and find the maximum
    numbers = [int(f.split('_')[-1].split('.')[0]) for f in polygon_files]
    return max(numbers) + 1

def update_spectrum_plot(polygon_num, points, color):
    """Update an existing spectrum plot or create a new one if it doesn't exist."""
    # Get the spectrum data
    r = np.array([p[1] for p in points])
    c = np.array([p[0] for p in points])
    rr, cc = polygon(r, c, cube.shape[:2])
    mask = np.zeros(cube.shape[:2], dtype=bool)
    mask[rr, cc] = True
    spectra = cube[mask, :]
    avg_spectrum = spectra.mean(axis=0)
    std_spectrum = spectra.std(axis=0)
    wavelengths = np.linspace(350, 1000, cube.shape[2])

    if polygon_num in spectrum_figs and plt.fignum_exists(spectrum_figs[polygon_num].number):
        # Update existing plot
        spectrum_fig = spectrum_figs[polygon_num]
        spectrum_ax = spectrum_fig.axes[0]
        spectrum_ax.clear()
        spectrum_ax.plot(wavelengths, avg_spectrum, label='Mean Reflectance', color=color)
        spectrum_ax.fill_between(wavelengths, avg_spectrum - std_spectrum, avg_spectrum + std_spectrum, 
                               alpha=0.3, color=color, label='Std Dev')
        spectrum_ax.set_title(f"Average Reflectance Spectrum for Polygon {polygon_num}")
        spectrum_ax.set_xlabel("Wavelength (nm)")
        spectrum_ax.set_ylabel("Reflectance")
        spectrum_ax.legend()
        spectrum_ax.grid(True)
        spectrum_fig.canvas.draw()
    else:
        # Create new plot
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
        spectrum_fig.canvas.manager.set_window_title(f"Polygon {polygon_num} Spectrum")
        spectrum_figs[polygon_num] = spectrum_fig
        spectrum_fig.canvas.draw()
        plt.show(block=False)

def update_polygon_data(polygon_num, points, color):
    """Update polygon data and regenerate spectrum."""
    # Update polygon data
    all_polygons[polygon_num] = {
        'points': points,
        'color': color
    }
    
    # Save updated polygon coordinates
    polygon_data = np.column_stack(([p[0] / cube.shape[1] for p in points], 
                                  [p[1] / cube.shape[0] for p in points]))
    polygon_path = f'{output_dir}/{args.filename}_polygon_{polygon_num}.csv'
    np.savetxt(polygon_path, polygon_data, delimiter=',', header='X_coord,Y_coord', comments='')
    
    # Update spectrum plot
    update_spectrum_plot(polygon_num, points, color)
    
    # Save spectrum data
    wavelengths = np.linspace(350, 1000, cube.shape[2])
    r = np.array([p[1] for p in points])
    c = np.array([p[0] for p in points])
    rr, cc = polygon(r, c, cube.shape[:2])
    mask = np.zeros(cube.shape[:2], dtype=bool)
    mask[rr, cc] = True
    spectra = cube[mask, :]
    avg_spectrum = spectra.mean(axis=0)
    std_spectrum = spectra.std(axis=0)
    
    # Save spectrum data for the whole polygon (mean + standard deviation)
    output_data = np.column_stack((wavelengths, avg_spectrum, std_spectrum))
    output_path = f'{output_dir}/{args.filename}_spectrum_polygon_{polygon_num}_summary.csv'
    np.savetxt(output_path, output_data, delimiter=',', header='Wavelength (nm),Mean Reflectance,Std Dev', comments='')
    
    # Save coordinates and spectrum data for the subsample (random 100 points)
    n_samples = min(100, len(spectra))
    if n_samples < len(spectra):
        sample_indices = np.random.choice(len(spectra), n_samples, replace=False)
        subsample = spectra[sample_indices]
        y_coords, x_coords = np.where(mask)
        coords = np.column_stack((x_coords, y_coords))
        subsample_coords = coords[sample_indices]
    else:
        subsample = spectra
        y_coords, x_coords = np.where(mask)
        coords = np.column_stack((x_coords, y_coords))
        subsample_coords = coords

    header = 'X_coord,Y_coord,' + ','.join([f'{w:.1f}' for w in wavelengths])
    subsample_data = np.column_stack((subsample_coords, subsample))
    subsample_path = f'{output_dir}/{args.filename}_spectrum_polygon_{polygon_num}_random_sample.csv'
    fmt = ['%d', '%d'] + ['%.6e'] * subsample.shape[1]
    np.savetxt(subsample_path, subsample_data, delimiter=',', header=header, comments='', fmt=fmt)

def redraw_all_polygons(current_points=None, current_color=None):
    """Redraw all polygons with their vertices and edges.
    
    Args:
        current_points: Optional list of points for the polygon being drawn
        current_color: Optional color for the current polygon
    """
    # Clear only the lines and points, not the text labels
    for artist in ax.lines + ax.collections:
        artist.remove()
    
    # Remove existing text labels
    for artist in ax.texts:
        artist.remove()
    
    # Redraw all processed polygons
    for poly_num, poly_data in all_polygons.items():
        points = poly_data['points']
        color = poly_data['color']
        # Draw vertices
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        ax.plot(x_coords, y_coords, 'o', color=color)
        # Draw edges
        if len(points) > 1:
            for i in range(len(points)):
                ax.plot([points[i][0], points[(i + 1) % len(points)][0]],
                       [points[i][1], points[(i + 1) % len(points)][1]], '-', color=color)
        
        # Calculate and plot the centroid with the label
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        ax.text(centroid_x, centroid_y, str(poly_num), 
                color=color, fontsize=12, fontweight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Draw current polygon if it exists
    if current_points is not None and len(current_points) > 0 and current_color is not None:
        x_coords = [p[0] for p in current_points]
        y_coords = [p[1] for p in current_points]
        ax.plot(x_coords, y_coords, 'o', color=current_color)
        if len(current_points) > 1:
            for i in range(len(current_points)):
                ax.plot([current_points[i][0], current_points[(i + 1) % len(current_points)][0]],
                       [current_points[i][1], current_points[(i + 1) % len(current_points)][1]], '-', color=current_color)
    
    fig.canvas.draw_idle()

def continue_to_polygon(event):
    global final_rgb, current_points, drawing_polygon, dragging_vertex, current_polygon, current_polygon_num, all_polygons, spectrum_figs
    low = low_slider.val
    high = high_slider.val
    gain = gain_slider.val
    offset = offset_slider.val
    p_low, p_high = np.percentile(rgb_raw, (low, high))
    final_rgb = np.clip((rgb_raw - p_low) / (p_high - p_low), 0, 1)
    final_rgb = np.clip(gain * final_rgb + offset, 0, 1)

    # Instead of clearing the axis, just update the image
    ax.images[0].set_data(final_rgb)
    ax.set_title("Draw polygons (Click to add points, press Enter to finish polygon, 'q' to quit)")
    fig.canvas.draw_idle()

    # Initialize or preserve current points and state
    current_points = []
    drawing_polygon = True
    dragging_vertex = None
    current_polygon = None
    current_polygon_num = None
    
    # Initialize dictionaries if they don't exist
    if all_polygons is None:
        all_polygons = {}
    if spectrum_figs is None:
        spectrum_figs = {}

    # Create a colormap with 10 distinct colors from hsv
    colors = plt.cm.hsv(np.linspace(0, 1, 10))
    # Create a list to track used colors, initialized with colors from existing polygons
    used_colors = []
    if all_polygons:
        # Get the maximum polygon number to determine how many colors are already used
        max_poly_num = max(all_polygons.keys())
        # Initialize used_colors with colors for existing polygons
        for i in range(max_poly_num):
            used_colors.append(colors[i % len(colors)])
    
    # Shuffle the remaining colors for new polygons
    remaining_colors = colors[len(used_colors):]
    np.random.shuffle(remaining_colors)
    
    # Create the final colors array
    if used_colors:
        colors = np.concatenate([np.array(used_colors), remaining_colors])
    else:
        colors = remaining_colors

    def get_color_for_polygon(polygon_num):
        # Get the color for a specific polygon number, ensuring consistency
        if polygon_num > len(used_colors):
            # If we haven't used all colors, pick a new one
            color = colors[(polygon_num - 1) % len(colors)]
            used_colors.append(color)
        else:
            # If we've used this number before, return the existing color
            color = used_colors[polygon_num - 1]
        return color

    def find_nearest_vertex(x, y, threshold=5):
        # Find the nearest vertex within threshold distance across all polygons
        nearest = None
        min_dist = threshold
        target_polygon = None
        target_polygon_num = None

        # Check current points first
        if current_points:
            distances = [(i, np.sqrt((p[0] - x)**2 + (p[1] - y)**2)) for i, p in enumerate(current_points)]
            idx, dist = min(distances, key=lambda x: x[1])
            if dist < min_dist:
                min_dist = dist
                nearest = idx
                target_polygon = current_points

        # Then check all processed polygons
        for poly_num, poly_data in all_polygons.items():
            points = poly_data['points']
            distances = [(i, np.sqrt((p[0] - x)**2 + (p[1] - y)**2)) for i, p in enumerate(points)]
            idx, dist = min(distances, key=lambda x: x[1])
            if dist < min_dist:
                min_dist = dist
                nearest = idx
                target_polygon = points
                target_polygon_num = poly_num

        return nearest, target_polygon, target_polygon_num

    def on_click(event):
        global dragging_vertex, current_polygon, current_polygon_num
        if event.inaxes != ax or not drawing_polygon:
            return

        # Check if we're clicking near any vertex
        vertex_idx, target_polygon, target_polygon_num = find_nearest_vertex(event.xdata, event.ydata)
        if vertex_idx is not None:
            dragging_vertex = vertex_idx
            current_polygon = target_polygon
            current_polygon_num = target_polygon_num
            return

        # If not dragging, add a new point
        current_points.append((event.xdata, event.ydata))
        # Get the next polygon number
        polygon_num = get_next_polygon_number(output_dir, args)
        # Get the color for this polygon number
        color = get_color_for_polygon(polygon_num)
        redraw_all_polygons(current_points, color)

    def on_motion(event):
        global dragging_vertex, current_polygon, current_polygon_num
        if event.inaxes != ax or not drawing_polygon or dragging_vertex is None:
            return

        # Update the vertex position
        if current_polygon_num is None:
            # Dragging vertex in current polygon
            current_points[dragging_vertex] = (event.xdata, event.ydata)
            color = get_color_for_polygon(get_next_polygon_number(output_dir, args))
            redraw_all_polygons(current_points, color)
        elif current_polygon_num in all_polygons:  # Only update if polygon exists
            # Dragging vertex in processed polygon
            current_polygon[dragging_vertex] = (event.xdata, event.ydata)
            color = all_polygons[current_polygon_num]['color']
            # Update the polygon data
            all_polygons[current_polygon_num]['points'] = current_polygon.copy()
            redraw_all_polygons()
        else:
            # If we somehow got here with an invalid polygon number, reset the state
            dragging_vertex = None
            current_polygon = None
            current_polygon_num = None

    def on_release(event):
        global dragging_vertex, current_polygon, current_polygon_num
        if dragging_vertex is not None and current_polygon_num is not None and current_polygon_num in all_polygons:
            # Update the processed polygon data and regenerate spectrum
            update_polygon_data(current_polygon_num, current_polygon, all_polygons[current_polygon_num]['color'])
            print(f"\nUpdated polygon {current_polygon_num} and saved changes to CSV files.")
        
        dragging_vertex = None
        current_polygon = None
        current_polygon_num = None

    def on_key(event):
        global drawing_polygon, current_points
        if event.key == 'enter' and current_points:
            # Get the next polygon number
            polygon_num = get_next_polygon_number(output_dir, args)
            # Get the color for this polygon number
            color = get_color_for_polygon(polygon_num)
            
            # Draw the closing edge from last point to first point
            if len(current_points) > 2:  # Only draw closing edge if we have at least 3 points
                ax.plot([current_points[-1][0], current_points[0][0]], 
                       [current_points[-1][1], current_points[0][1]], '-', color=color)
                fig.canvas.draw_idle()
            
            print(f"\nProcessing polygon {polygon_num} with {len(current_points)} points")
            # Store the current points before clearing
            pts_to_process = current_points.copy()
            
            # Add to all_polygons dictionary
            all_polygons[polygon_num] = {
                'points': pts_to_process,
                'color': color
            }
            
            # Process the stored points and save data
            update_polygon_data(polygon_num, pts_to_process, color)
            
            # Clear current points and redraw all polygons
            current_points.clear()
            redraw_all_polygons()
        elif event.key == 'q':
            # Process any remaining points before quitting
            if current_points:
                # Get the next polygon number
                polygon_num = get_next_polygon_number(output_dir, args)
                # Get the color for this polygon number
                color = get_color_for_polygon(polygon_num)
                
                # Draw the closing edge from last point to first point
                if len(current_points) > 2:  # Only draw closing edge if we have at least 3 points
                    ax.plot([current_points[-1][0], current_points[0][0]], 
                           [current_points[-1][1], current_points[0][1]], '-', color=color)
                    fig.canvas.draw_idle()
                
                print(f"\nProcessing final polygon {polygon_num} with {len(current_points)} points")
                pts_to_process = current_points.copy()
                
                # Add to all_polygons dictionary
                all_polygons[polygon_num] = {
                    'points': pts_to_process,
                    'color': color
                }
                
                update_polygon_data(polygon_num, pts_to_process, color)
                current_points.clear()
                redraw_all_polygons()
            print(f"\nTotal polygons processed: {len(all_polygons)}")
            drawing_polygon = False
            plt.disconnect(cid_click)
            plt.disconnect(cid_key)
            plt.disconnect(cid_motion)
            plt.disconnect(cid_release)

    # Connect event handlers
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
    print("Draw your polygons. Click to add points, press Enter to finish each polygon, 'q' to quit.")
    print("You can drag vertices to adjust their positions, even after processing.")

def load_polygons(event):
    # Load existing polygon coordinates from CSV files and draw them on the image
    global all_pts, drawing_polygon, final_rgb, all_polygons, spectrum_figs, dragging_vertex, current_polygon, current_polygon_num
    global cid_click, cid_motion, cid_release
    
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
        all_polygons = {}  # Dictionary to store all polygons and their data
        spectrum_figs = {}  # Dictionary to store spectrum plot windows
        dragging_vertex = None
        current_polygon = None
        current_polygon_num = None
        drawing_polygon = True  # Enable polygon editing
        
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
            
            # Store polygon data in all_polygons dictionary
            all_polygons[polygon_num] = {
                'points': pts,
                'color': used_colors[polygon_num - 1]
            }
            
            # Draw the polygon
            color = used_colors[polygon_num - 1]
            # Draw vertices
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            ax.plot(x_coords, y_coords, 'o', color=color)
            # Draw edges
            if len(pts) > 1:
                for i in range(len(pts)):
                    ax.plot([pts[i][0], pts[(i + 1) % len(pts)][0]],
                           [pts[i][1], pts[(i + 1) % len(pts)][1]], '-', color=color)
            
            # Calculate and plot the centroid with the label
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            ax.text(centroid_x, centroid_y, str(polygon_num), 
                    color=color, fontsize=12, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'))
            
            print(f"Finished processing polygon {polygon_num}")
        
        fig.canvas.draw_idle()
        print(f"\nLoaded {len(all_pts)} polygons from CSV files.")
        
        # Connect event handlers for dragging
        def find_nearest_vertex(x, y, threshold=5):
            # Find the nearest vertex within threshold distance across all polygons
            nearest = None
            min_dist = threshold
            target_polygon = None
            target_polygon_num = None

            # Check all processed polygons
            for poly_num, poly_data in all_polygons.items():
                points = poly_data['points']
                distances = [(i, np.sqrt((p[0] - x)**2 + (p[1] - y)**2)) for i, p in enumerate(points)]
                idx, dist = min(distances, key=lambda x: x[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest = idx
                    target_polygon = points
                    target_polygon_num = poly_num

            return nearest, target_polygon, target_polygon_num

        def redraw_all_polygons():
            # Redraw all polygons with their vertices and edges
            # Clear only the lines and points, not the text labels
            for artist in ax.lines + ax.collections:
                artist.remove()
            
            # Remove existing text labels
            for artist in ax.texts:
                artist.remove()
            
            # Redraw all processed polygons
            for poly_num, poly_data in all_polygons.items():
                points = poly_data['points']
                color = poly_data['color']
                # Draw vertices
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                ax.plot(x_coords, y_coords, 'o', color=color)
                # Draw edges
                if len(points) > 1:
                    for i in range(len(points)):
                        ax.plot([points[i][0], points[(i + 1) % len(points)][0]],
                               [points[i][1], points[(i + 1) % len(points)][1]], '-', color=color)
            
                # Calculate and plot the centroid with the label
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                ax.text(centroid_x, centroid_y, str(poly_num), 
                        color=color, fontsize=12, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.3'))
            
            fig.canvas.draw_idle()

        def on_click(event):
            global dragging_vertex, current_polygon, current_polygon_num
            if event.inaxes != ax or not drawing_polygon:
                return

            # Check if we're clicking near any vertex
            vertex_idx, target_polygon, target_polygon_num = find_nearest_vertex(event.xdata, event.ydata)
            if vertex_idx is not None:
                dragging_vertex = vertex_idx
                current_polygon = target_polygon
                current_polygon_num = target_polygon_num
                return

        def on_motion(event):
            global dragging_vertex, current_polygon, current_polygon_num
            if event.inaxes != ax or not drawing_polygon or dragging_vertex is None:
                return

            # Update the vertex position
            current_polygon[dragging_vertex] = (event.xdata, event.ydata)
            color = all_polygons[current_polygon_num]['color']
            # Update the polygon data
            all_polygons[current_polygon_num]['points'] = current_polygon.copy()

            # Redraw all polygons
            redraw_all_polygons()

        def on_release(event):
            global dragging_vertex, current_polygon, current_polygon_num
            if dragging_vertex is not None and current_polygon_num is not None:
                # Update the processed polygon data and regenerate spectrum
                update_polygon_data(current_polygon_num, current_polygon, all_polygons[current_polygon_num]['color'])
                print(f"\nUpdated polygon {current_polygon_num} and saved changes to CSV files.")
            
            dragging_vertex = None
            current_polygon = None
            current_polygon_num = None

        # Connect event handlers
        cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
        cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
        cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
        
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
            for poly_num, poly_data in all_polygons.items():
                update_spectrum_plot(poly_num, poly_data['points'], poly_data['color'])
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

# Connect the radio buttons
vis_method.on_clicked(change_vis_method)

plt.show()
