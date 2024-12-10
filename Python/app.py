import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np


# Preprocessing Module
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    resized_image = cv2.resize(image, (256, 256))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    enhanced_image = cv2.equalizeHist(blurred_image)
    return enhanced_image


# Feature Extraction Module
def extract_morphological_features(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return {
        "Erosion": erosion,
        "Dilation": dilation,
        "Opening": opening,
        "Closing": closing
    }


# Similarity Measurement Module
def calculate_featurewise_similarity(query_features_dict, dataset_features_dict):
    featurewise_similarities = {}
    for image_name, dataset_features in dataset_features_dict.items():
        similarities = {}
        for feature_name in query_features_dict.keys():
            distance = np.linalg.norm(
                query_features_dict[feature_name].flatten() - dataset_features[feature_name].flatten()
            )
            similarities[feature_name] = distance
        featurewise_similarities[image_name] = sorted(similarities.items(), key=lambda x: x[1])
    return featurewise_similarities


# GUI Application
class MorphologicalMatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Morphological Matching")
        self.root.geometry("800x600")

        self.query_image_path = None
        self.dataset_image_paths = []
        self.dataset_features_dict = {}
        self.query_features_dict = {}

        # Buttons
        tk.Button(root, text="Upload Query Image", command=self.upload_query_image).pack(pady=10)
        tk.Button(root, text="Upload Dataset Images", command=self.upload_dataset_images).pack(pady=10)
        tk.Button(root, text="Calculate Similarity", command=self.calculate_similarity).pack(pady=10)

        # Canvas for displaying query image
        self.query_canvas = tk.Canvas(root, width=256, height=256, bg="gray")
        self.query_canvas.pack(pady=10)

        # Scrollable results frame
        self.results_frame_container = tk.Frame(root)
        self.results_frame_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # Add a canvas and scrollbar to the container
        self.results_canvas = tk.Canvas(self.results_frame_container)
        self.scrollbar = tk.Scrollbar(self.results_frame_container, orient=tk.VERTICAL, command=self.results_canvas.yview)
        self.scrollable_frame = tk.Frame(self.results_canvas)

        # Configure the canvas and scrollbar
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def upload_query_image(self):
        self.query_image_path = filedialog.askopenfilename(title="Select Query Image")
        if self.query_image_path:
            preprocessed_query = preprocess_image(self.query_image_path)
            self.query_features_dict = extract_morphological_features(preprocessed_query)
            self.display_image(self.query_canvas, self.query_image_path)
            messagebox.showinfo("Success", "Query image uploaded and processed successfully!")

    def upload_dataset_images(self):
        self.dataset_image_paths = filedialog.askopenfilenames(title="Select Dataset Images")
        if self.dataset_image_paths:
            self.dataset_features_dict = {}
            for image_path in self.dataset_image_paths:
                preprocessed_image = preprocess_image(image_path)
                self.dataset_features_dict[image_path] = extract_morphological_features(preprocessed_image)
            messagebox.showinfo("Success", "Dataset images uploaded and processed successfully!")

    def calculate_similarity(self):
        if not self.query_features_dict or not self.dataset_features_dict:
            messagebox.showerror("Error", "Please upload both a query image and dataset images.")
            return

        featurewise_results = calculate_featurewise_similarity(self.query_features_dict, self.dataset_features_dict)
        self.display_results(featurewise_results)

    def display_image(self, canvas, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

    def display_results(self, featurewise_results):
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for image_name, similarities in featurewise_results.items():
            result_text = f"Results for {image_name}:\n" + "\n".join(
                [f"{feature_name}: {similarity_score:.2f}" for feature_name, similarity_score in similarities]
            )

            # Create a label for the result text
            result_label = tk.Label(self.scrollable_frame, text=result_text, justify="left")
            result_label.pack(pady=5)

            # Create a canvas to display the image for each result
            result_image = cv2.imread(image_name)
            result_image = cv2.cvtColor(cv2.resize(result_image, (256, 256)), cv2.COLOR_BGR2RGB)
            result_image = ImageTk.PhotoImage(Image.fromarray(result_image))

            # Display the image in a canvas
            result_canvas = tk.Canvas(self.scrollable_frame, width=256, height=256, bg="gray")
            result_canvas.create_image(0, 0, anchor=tk.NW, image=result_image)
            result_canvas.image = result_image
            result_canvas.pack(pady=5)

            # Display the image path below the image
            path_label = tk.Label(self.scrollable_frame, text=image_name, justify="left")
            path_label.pack(pady=5)


# Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = MorphologicalMatchingApp(root)
    root.mainloop()
