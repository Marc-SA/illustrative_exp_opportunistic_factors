import cv2
import numpy as np

def calculate_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def calculate_accuracy(distances, ground_truth, threshold):
    correct = 0
    total = 0
    for i in range(len(distances)):
        for j in range(len(distances[i])):
            if ground_truth[i][j] == 1:  # Same vehicle in ground truth
                total += 1
                if distances[i][j] < threshold:
                    correct += 1
            elif ground_truth[i][j] == 0:  # Different vehicles in ground truth
                total += 1
                if distances[i][j] >= threshold:
                    correct += 1
    return correct / total if total > 0 else 0

def main():
    car_paths = {
        'car1': ['Car1_1.png', 'Car1_2.png', 'Car1_3.png'],
        'car2': ['Car2_1.png', 'Car2_2.png', 'Car2_3.png'],
        'car3': ['Car3_1.png', 'Car3_2.png', 'Car3_3.png'],
        'car5': ['Car5_1.png', 'Car5_2.png', 'Car5_3.png']
        # Add paths for  other cars and images if necessary
    }

    num_images_per_car = len(next(iter(car_paths.values())))  # Assuming all cars have the same number of images (samles)

    # matrix to store similarity results
    similarity_matrix = np.zeros((len(car_paths), len(car_paths)))

    # Ground truth: Define which images belong to the same (1) or different (0) vehicles
    ground_truth = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    # Calculate histograms for each image of each car and fill in the similarity matrix
    for i, (car1, image_paths1) in enumerate(car_paths.items()):
        for j, (car2, image_paths2) in enumerate(car_paths.items()):
            similarities = []
            for img1_index, image_path1 in enumerate(image_paths1):
                for img2_index, image_path2 in enumerate(image_paths2):
                    if car1 != car2 or img1_index != img2_index:
                        image1 = cv2.imread(image_path1)
                        image2 = cv2.imread(image_path2)
                        if image1 is None or image2 is None:
                            print(f"Error: Unable to load images {image_path1} or {image_path2}")
                            continue
                        hist1 = calculate_histogram(image1)
                        hist2 = calculate_histogram(image2)
                        similarity = compare_histograms(hist1, hist2)
                        similarities.append(similarity)
            if similarities:
                mean_similarity = np.mean(similarities)
                similarity_matrix[i, j] = mean_similarity

    # Calculate accuracy using the threshold of 0.2
    threshold = 0.2
    accuracy = calculate_accuracy(similarity_matrix, ground_truth, threshold)

    print(f"Accuracy at threshold {threshold}: {accuracy:.4f}")
    print(similarity_matrix)

if __name__ == "__main__":
    main()

