from flask import Flask, request, render_template, Response, send_file
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import splprep, splev
import cv2

app = Flask(__name__)

# ... (Your existing `read_csv` function remains unchanged)

# ... (Your existing `generate_svg` function remains unchanged)

# ... (Implement the core curve analysis functions: `regularize_curves`, `detect_symmetry`, `complete_curves`)

# Function to find contours and symmetry lines using OpenCV
def find_contours_and_symmetry_lines(paths_XYs, img_size=(512, 512)):
    """Find contours and symmetry lines using OpenCV."""
    symmetry_lines = []
    for path in paths_XYs:
        img = np.zeros(img_size, dtype=np.uint8)
        for polyline in path:
            points = polyline.astype(np.int32)
            cv2.polylines(img, [points], isClosed=False, color=255, thickness=2)

        binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)
            center, axes, angle = ellipse
            p1 = (center[0] + axes[0] / 2 * np.cos(np.deg2rad(angle)),
                  center[1] + axes[0] / 2 * np.sin(np.deg2rad(angle)))
            p2 = (center[0] - axes[0] / 2 * np.cos(np.deg2rad(angle)),
                  center[1] - axes[0] / 2 * np.sin(np.deg2rad(angle)))
            symmetry_lines.append((p1, p2))
    return symmetry_lines

# Function to plot and save the plot as an image
def plot_and_save(paths_XYs, title, filename, symmetry_lines=None):
    """Plot the polylines and save the plot as an image."""
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)

    # Plot symmetry lines if provided
    if symmetry_lines:
        for line in symmetry_lines:
            p1, p2 = line
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', linewidth=1)

    # Save the plot
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to avoid memory issues

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            # Read the CSV file
            paths_XYs = read_csv(file)

            # Process the curves
            regularized_paths = regularize_curves(paths_XYs)
            symmetry_lines = find_contours_and_symmetry_lines(paths_XYs)
            completed_paths = complete_curves(paths_XYs)

            # Generate plots and save them
            plot_and_save(paths_XYs, 'Original Curves', 'original_plot.png')
            plot_and_save(regularized_paths, 'Regularized Curves', 'regularized_plot.png')
            plot_and_save(paths_XYs, 'Symmetric Curves with Lines', 'symmetry_plot.png', symmetry_lines)
            plot_and_save(completed_paths, 'Completed Curves', 'completed_plot.png')

            # Save the processed data to a new CSV file
            output_csv = 'processed_curves.csv'
            write_csv(completed_paths, output_csv)

            # Render the results template
            return render_template('results.html', 
                                   original_plot='original_plot.png',
                                   regularized_plot='regularized_plot.png',
                                   symmetry_plot='symmetry_plot.png',
                                   completed_plot='completed_plot.png',
                                   output_csv=output_csv)

    else:
        return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
