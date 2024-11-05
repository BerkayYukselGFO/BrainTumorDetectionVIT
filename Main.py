import os
import cv2
import torch
import random
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Kayıt klasörünü tanımlayın
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# 1. Veri yükleme fonksiyonu (Yeni sınıf isimlerine göre güncellendi)
def load_data(data_dir):
    categories = ['glioma', 'meningioma', 'pituitary', 'notumor']
    data = []
    for idx, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        label = idx  # Her kategoriye bir sayı etiketi veriyoruz: 0, 1, 2, 3
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                data.append((img, label))
    return data

# 2. Görüntüleri yeniden boyutlandırma fonksiyonu
def resize_images(data, target_size=(224, 224)):
    resized_data = []
    for img, label in data:
        img_resized = cv2.resize(img, target_size)
        resized_data.append((img_resized, label))
    return resized_data

# 3. Veri kümesini bölme fonksiyonu
def split_data(data, test_split=0.2):
    images, labels = zip(*data)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_split, random_state=42)
    return train_images, train_labels, test_images, test_labels

# Görsel olarak veri örnekleri gösterme
def plot_samples(data, labels, category_name, n=5):
    plt.figure(figsize=(15, 3))
    category_data = [img for img, label in zip(data, labels) if label == category_name]
    samples = random.sample(category_data, n)
    for i, img in enumerate(samples):
        plt.subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.suptitle(f"Sample images for category: {category_name}")
    plt.savefig(os.path.join(output_dir, f"sample_images_category_{category_name}.png"))
    plt.close()

# Eğitim süreci kayıp ve doğruluk görselleştirme
def plot_training_history(train_losses, val_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Eğitim Kaybı')
    plt.plot(epochs, val_losses, 'r', label='Doğrulama Kaybı')
    plt.xlabel('Epochs')
    plt.ylabel('Kayıp')
    plt.legend()
    plt.title('Eğitim ve Doğrulama Kaybı')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'g', label='Doğruluk')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk')
    plt.legend()
    plt.title('Doğruluk')

    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

# Modeli ve özellik çıkarıcıyı yükleme (4 sınıfa göre ayarlama)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri seti sınıfı
class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Eğitim fonksiyonu
def train_model(train_data, train_labels, val_data, val_labels, epochs=5, batch_size=16, lr=2e-5):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])

    train_dataset = BrainTumorDataset(train_data, train_labels, transform=transform)
    val_dataset = BrainTumorDataset(val_data, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_losses, accuracies = [], [], []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Doğrulama
        model.eval()
        total_val_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / len(val_dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Eğitim Kaybı: {avg_train_loss:.4f}")
        print(f"  Doğrulama Kaybı: {avg_val_loss:.4f}")
        print(f"  Doğruluk: {accuracy:.4f}")

    plot_training_history(train_losses, val_losses, accuracies)

# Test sonuçlarını görselleştirme
def evaluate_model(model, test_loader, class_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(classification_report(y_true, y_pred, target_names=class_names))

# Ana fonksiyon
def main():
    data_dir_train = 'Training'
    data_dir_test = 'Testing'
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]

    train_data = load_data(data_dir_train)
    test_data = load_data(data_dir_test)

    resized_train_data = resize_images(train_data)
    resized_test_data = resize_images(test_data)

    train_images, train_labels, val_images, val_labels = split_data(resized_train_data, test_split=0.2)
    test_images, test_labels = zip(*resized_test_data)

    plot_samples(train_images, train_labels, category_name=0)  # Glioma örnekleri
    plot_samples(train_images, train_labels, category_name=1)  # Meningioma örnekleri

    train_model(train_images, train_labels, val_images, val_labels, epochs=5)
    
    test_dataset = BrainTumorDataset(test_images, test_labels, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    evaluate_model(model, test_loader, class_names)

if __name__ == "__main__":
    main()
