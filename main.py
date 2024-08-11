from flask import Flask, request, render_template, Response, send_file
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import splprep, splev
import cv2
PIXELS_TO_MM = 0.2645833333

# Regularize curves using Douglas-Peucker
def regularize_curve(curve, epsilon=5.0):
    curve = np.array(curve, dtype=np.float32)
    if len(curve) >= 2:
        return cv2.approxPolyDP(curve, epsilon, closed=False)[:, 0, :].tolist()
    return curve

# Function to detect shapes
def get_shapes(img, imgContour):
    shapes = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(imgContour, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(imgContour, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            if len(approx) == 3:
                shape_type = "Triangle"
            elif len(approx) == 4:
                asp_ratio = w / float(h)
                shape_type = "Square" if 0.9 <= asp_ratio <= 1.1 else "Rectangle"
            elif len(approx) == 5:
                shape_type = "Pentagon"
            elif len(approx) > 5:
                shape_type = "Polygon"
            else:
                shape_type = "Unknown"
            
            shapes.append({
                'type': shape_type,
                'contour': approx
            })
            cv2.drawContours(imgContour, [approx], -1, (0, 255, 0), 2)
            cv2.putText(imgContour, shape_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return shapes

# Function to detect symmetry
def check_symmetry(shape_mask):
    flipped_h = cv2.flip(shape_mask, 1)
    flipped_v = cv2.flip(shape_mask, 0)
    flipped_d1 = cv2.transpose(cv2.flip(shape_mask, 0))
    flipped_d2 = cv2.transpose(cv2.flip(shape_mask, 1))

    diff_h = np.sum(np.abs(shape_mask - flipped_h))
    diff_v = np.sum(np.abs(shape_mask - flipped_v))
    diff_d1 = np.sum(np.abs(shape_mask - flipped_d1))
    diff_d2 = np.sum(np.abs(shape_mask - flipped_d2))

    return {
        'horizontal': diff_h == 0,
        'vertical': diff_v == 0,
        'diagonal1': diff_d1 == 0,
        'diagonal2': diff_d2 == 0
    }

# Function to complete curves
def complete_curve(curve, threshold=10.0):
    completed_curve = []
    for i in range(len(curve) - 1):
        completed_curve.append(curve[i])
        if np.linalg.norm(np.array(curve[i]) - np.array(curve[i + 1])) > threshold:
            mid_point = ((np.array(curve[i]) + np.array(curve[i + 1])) / 2).astype(int)
            completed_curve.append(mid_point.tolist())
    completed_curve.append(curve[-1])
    return completed_curve

# Read CSV file
def read_csv(csv_path):
    paths_XYs = []
    with open(csv_path, 'r') as file:
        for line in file:
            points = []
            for p in line.strip().split():
                try:
                    points.append(tuple(map(float, p.split(','))))
                except ValueError:
                    continue
            if points:
                paths_XYs.append(np.array(points))
    return paths_XYs

# Function to plot shapes
def plot(paths_XYs, title='Curves'):
    fig, ax = plt.subplots(figsize=(8, 8))
    for path in paths_XYs:
        try:
            x, y = zip(*path)
            ax.plot(x, y, marker='o')
        except ValueError:
            continue
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.show()

# Convert polylines to image
def paths_to_image(paths_XYs, img_size=(500, 500)):
    img = np.ones((*img_size, 3), dtype=np.uint8) * 255
    for path in paths_XYs:
        for i in range(len(path) - 1):
            cv2.line(img, (int(path[i][0]), int(path[i][1])), (int(path[i + 1][0]), int(path[i + 1][1])), (0, 0, 0), 1)
    return img

# Process the curves
def process_curves(input_csv, output_csv, output_svg, output_png):
    paths_XYs = read_csv(input_csv)
    
    # Convert paths to image
    img = paths_to_image(paths_XYs)
    imgContour = img.copy()

    # Detect shapes
    shapes = get_shapes(img, imgContour)

    # Regularize curves
    regularized_paths = [regularize_curve(path) for path in paths_XYs]
    plot(regularized_paths, title='Regularized Curves')

    # Check symmetry
    for shape in shapes:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [shape['contour']], -1, 255, thickness=cv2.FILLED)
        symmetry = check_symmetry(mask)
        shape_type = shape['type']
        for axis, is_symmetric in symmetry.items():
            print(f"{shape_type} is {'symmetric' if is_symmetric else 'not symmetric'} along the {axis} axis.")

    # Complete curves
    completed_paths = [complete_curve(path) for path in regularized_paths]
    plot(completed_paths, title='Completed Curves')

    # Save completed paths to CSV
    with open(output_csv, 'w') as f:
        for path in completed_paths:
            for point in path:
                f.write(f"{point[0]},{point[1]}\n")
            f.write('\n')

    # Save completed paths as SVG
    paths_to_svg(completed_paths, output_svg)
    
    # Convert SVG to PNG
    svg_to_png(output_svg, output_png)

# Example usage
input_csv = 'isolated.csv'
output_csv = 'isolated_sol.csv'
output_svg = 'solution.svg'
output_png = 'solution.png'

process_curves(input_csv, output_csv, output_svg, output_png)