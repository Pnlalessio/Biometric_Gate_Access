from flask import Flask, request, render_template, jsonify
import numpy as np
from facenet_pytorch import MTCNN
import os
import cv2
import warnings
import torch
from PIL import Image
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from skimage.feature import local_binary_pattern
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

warnings.filterwarnings("ignore")

# Directories for storing images and global variables to track registration and verification process
app = Flask(__name__)
UPLOAD_FOLDER = 'static'
IDENTITY_FOLDER = 'static'
VERIFICATION_FOLDER = 'static'
ANDROID_ID = ""
ANDROID_ID_VERIFICATION = ""
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False)

# Set image size and parameters for Local Binary Pattern (LBP)
number_of_verification_antispoof_probes = 0
minimum_number_of_matches = 0
IMG_SIZE = (224, 224)
radius = 1
n_points = 8 * radius

# Function to apply Local Binary Pattern (LBP) on a single channel
def apply_lbp(channel):
    return local_binary_pattern(channel, n_points, radius, method='uniform')

# Function to apply LBP on an RGB image
def lbp_transform(image):
    lbp_r = apply_lbp(image[:, :, 0]) # Apply LBP on the red channel
    lbp_g = apply_lbp(image[:, :, 1]) # Apply LBP on the green channel
    lbp_b = apply_lbp(image[:, :, 2]) # Apply LBP on the blue channel
    # Normalize and merge the three channels
    lbp_r_normalized = cv2.normalize(lbp_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    lbp_g_normalized = cv2.normalize(lbp_g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    lbp_b_normalized = cv2.normalize(lbp_b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    lbp_color = cv2.merge((lbp_r_normalized, lbp_g_normalized, lbp_b_normalized))
    return lbp_color

# Load our pre-trained MobileNet models for spoof detection (MobileNet or MobileNet + LBP)
def load_trained_mobilenet_model(model_path, device):
    loaded_model = models.mobilenet_v2(pretrained=False)
    loaded_model.classifier[1] = nn.Linear(loaded_model.classifier[1].in_features, 1)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    return loaded_model

# Preprocess an image (resize, normalize, convert to tensor)
def preprocess_image(image, target_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Detect whether the face in the image is real or spoofed using the loaded model
def detect_real_or_fake(image, model, target_size):
    model.eval()
    with torch.no_grad():
        processed_image = preprocess_image(image, target_size).to(device)
        output = model(processed_image)
        prediction = torch.sigmoid(output).item() # Binary prediction (Real or Spoof)
        return prediction, "Fake" if prediction > 0.50 else "Real"

# Load images from a folder, resize and apply transformations
def load_images_from_folder(folder, transform, img_size):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('png', 'jpg', 'jpeg', 'gif')) and "face_photo" in filename:
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            img = transform(img)
            images.append(img)
    return images

# Load the InceptionResnetV1 model in pytorch (pretrained on VGGFace2 and CASIA-Webface) for face embeddings
def load_model(pretrained_on_dataset, device):
    resnet = InceptionResnetV1(
        classify=False,
        pretrained=pretrained_on_dataset
    ).to(device)
    return resnet

# Create embeddings for a list of images
def create_embeddings_for_images(model, images, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0).to(device)
            embedding = model(img)
            embeddings.append(embedding.squeeze(0).cpu())
    return torch.stack(embeddings).cuda()

# Calculate the Euclidean distance between a face embedding and registered embeddings
def calculate_distance(face_embedding, image_embeddings):
    face_embedding = torch.tensor(face_embedding).unsqueeze(0).cuda()
    distance_matrix = torch.cdist(face_embedding, image_embeddings, p=2).squeeze(0)
    return distance_matrix

# Remove temporary probe folder after verification
def remove_TEMP_PROBES_FOLDER(TEMP_PROBES_FOLDER):
    if os.path.exists(TEMP_PROBES_FOLDER):
        for filename in os.listdir(TEMP_PROBES_FOLDER):
            file_path = os.path.join(TEMP_PROBES_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error during the elimination of the file {file_path}: {e}")

        try:
            os.rmdir(TEMP_PROBES_FOLDER)
            print(f"The folder {TEMP_PROBES_FOLDER} has been deleted.")
        except Exception as e:
            print(f"Error during the elimination of the folder  {TEMP_PROBES_FOLDER}: {e}")
    else:
        print(f"The folder {TEMP_PROBES_FOLDER} doesn't exist.")

# Load one of the pre-trained MobileNet models for anti-spoofing
loaded_model = load_trained_mobilenet_model(r"C:\Users\Asus\Desktop\Progetto_BS_Frabotta_Ferrone\Evaluation\Face_antispoofing\Models\mobilenet_v2_combined_model.pth", device)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/registration', methods=['POST'])
def upload():
    images = []
    images_names = []
    android_id = request.form.get('android_id')

    # Check for the presence of multiple image files
    for i in range(3):
        if f'image{i}' in request.files:
            file = request.files[f'image{i}']
            if file.filename != '':
                # Read the image directly from the request
                image = np.fromstring(file.read(), np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                images.append(image)
                images_names.append(file.filename)
            else:
                return jsonify({'error': f'No selected file for image{i}'}), 400
        else:
            return jsonify({'error': f'No image{i} part in the request'}), 400

    global ANDROID_ID
    ANDROID_ID = android_id

    # Check if images list is empty
    if not images:
        return jsonify({'error': 'No images provided'}), 400

    # Create a directory for the android_id if it doesn't exist
    android_folder = os.path.join(UPLOAD_FOLDER, android_id)
    if not os.path.exists(android_folder):
        os.makedirs(android_folder)

    valid_faces = []

    # In the following lines, MTCNN is used for face detection to locate the bounding box positions of the face in the image.
    # This allows us to extract the region of interest around the face and crop the image according to the bounding box.
    for idx, image in enumerate(images):
        # Convert the image to RGB and process
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Detect face using MTCNN
        boxes, _ = mtcnn.detect(image_pil)
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])

                if x1 < x2 and y1 < y2:
                    face = image[y1:y2, x1:x2]

                    if face.size > 0:
                        valid_faces.append(face)
                    else:
                        return jsonify({
                            'error': f'Face cropping failed for image{idx}',
                            'status': 'failure'
                        }), 400
        else:
            return jsonify({
                'error': f'No face detected in image{idx}',
                'status': 'failure'
            }), 400

    # If all the images contain faces, save all the embeddings
    if len(valid_faces) == 3:
        for idx, face in enumerate(valid_faces):
            face = cv2.resize(face, IMG_SIZE)
            trans = transforms.Compose([
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
            ])

            face_tensor = trans(face).unsqueeze(0).cuda()
            resnet_model = load_model('vggface2', device)

            # Create the face embeddings
            resnet_model.eval()
            with torch.no_grad():
                face_embedding = resnet_model(face_tensor).squeeze(0).cpu().numpy()
            # Save the embeddings as a numpy array
            file_name_without_extension = os.path.splitext(images_names[idx])[0]

            embedding_filename = f"embedding_{file_name_without_extension}.npy"
            embedding_filepath = os.path.join(android_folder, embedding_filename)
            np.save(embedding_filepath, face_embedding)

        return jsonify({
            'message': 'All images processed successfully, embeddings saved',
            'status': 'success'
        }), 200
    else:
        return jsonify({
            'error': 'Failed to detect valid faces in all images',
            'status': 'failure'
        }), 400


# Route for face verification
@app.route('/verification', methods=['POST'])
def verification():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    android_id = request.form.get('android_id')
    number_of_verification_calls = request.form.get('number_of_verification_calls')

    global ANDROID_ID_VERIFICATION
    ANDROID_ID_VERIFICATION = android_id
    REGISTERED_USER_FOLDER = os.path.join('static', ANDROID_ID_VERIFICATION)
    global minimum_number_of_matches
    global number_of_verification_antispoof_probes

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not os.path.exists(REGISTERED_USER_FOLDER):
        return jsonify({'error': 'You are not registered'}), 400
    if file:
        if not os.listdir(REGISTERED_USER_FOLDER) :
            return jsonify({'error': 'Images not found for this Android ID'}), 400

        embedding_files = [file for file in os.listdir(REGISTERED_USER_FOLDER) if file.endswith('.npy')]
        TEMP_PROBES_FOLDER = os.path.join('static', 'TEMP_PROBES_FOLDER')
        if not os.path.exists(TEMP_PROBES_FOLDER):
            os.makedirs(TEMP_PROBES_FOLDER)

        # Temporarily saves the photo taken by the user
        filepath = os.path.join(TEMP_PROBES_FOLDER, file.filename)
        file.save(filepath)

        # Load the image and convert it to RGB
        image = cv2.imread(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        boxes, _ = mtcnn.detect(image_pil) # Detect faces using MTCNN
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])

                if x1 < x2 and y1 < y2:
                    face = image[y1:y2, x1:x2]

                    if face.size > 0:
                        face = cv2.resize(face, IMG_SIZE)
                        # Anti-spoof detection (check if the face extracted is real or spoof)
                        prediction, label_prediction = detect_real_or_fake(face, loaded_model, IMG_SIZE)
                        if prediction > 0.50:
                            print("Spoof")
                            # the face is fake
                            number_of_verification_antispoof_probes += 1
                            global minimum_number_of_matches
                            minimum_number_of_matches = 0
                            remove_TEMP_PROBES_FOLDER(TEMP_PROBES_FOLDER)
                            print("REJECTED")
                            return jsonify({'error': 'The face detected is spoof', 'value': number_of_verification_antispoof_probes}), 400
                        else:
                            print("Real")
                            trans = transforms.Compose([
                                np.float32,
                                transforms.ToTensor(),
                                fixed_image_standardization
                            ])

                            face_tensor = trans(face).unsqueeze(0).cuda()
                            resnet_model = load_model('vggface2', device)
                            resnet_model.eval()
                            with torch.no_grad():
                                face_embedding = resnet_model(face_tensor).squeeze(0).cpu()

                            embeddings = []
                            for embedding_file in embedding_files:
                                embedding_path = os.path.join(REGISTERED_USER_FOLDER, embedding_file)
                                embedding = np.load(embedding_path)
                                embeddings.append(torch.tensor(embedding).cuda())

                            image_embeddings = torch.stack(embeddings).cuda()
                            distance_matrix = calculate_distance(face_embedding, image_embeddings)

                            threshold = 0.58
                            min_distance = torch.min(distance_matrix)

                            if min_distance <= threshold:
                                result = 1
                                minimum_number_of_matches += result
                            else:
                                result = 0

                            print(f"The result of the matching is: {result}")
                            print(minimum_number_of_matches)
                            if minimum_number_of_matches == 3:
                                print("ACCEPTED")
                                minimum_number_of_matches = 0
                                remove_TEMP_PROBES_FOLDER(TEMP_PROBES_FOLDER)
                                return jsonify({
                                    'message': 'ACCEPTED'
                                }), 200

        if number_of_verification_calls == "3" and minimum_number_of_matches != 3:
            print("REJECTED")
            minimum_number_of_matches = 0
            remove_TEMP_PROBES_FOLDER(TEMP_PROBES_FOLDER)
            return jsonify({'error': 'Face not detected'}), 400

    if number_of_verification_calls == "3" and minimum_number_of_matches != 3:
        print("REJECTED")
        minimum_number_of_matches = 0
        return jsonify({'error': 'Image upload failed'}), 400

    return jsonify({
        'message': 'ACCEPTED'
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)








