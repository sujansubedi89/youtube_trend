import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

class ThumbnailCTRPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.img_size = (224, 224)
        
        # Load pre-trained face detector (free, included with OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def extract_thumbnail_features(self, img_path):
        """Extract visual features from thumbnail"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Resize
            img_resized = cv2.resize(img, self.img_size)
            
            # Convert to different color spaces
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            features = {}
            
            # 1. Color Analysis
            features['mean_brightness'] = np.mean(img_hsv[:, :, 2])
            features['mean_saturation'] = np.mean(img_hsv[:, :, 1])
            features['color_variance'] = np.var(img_rgb)
            
            # Dominant colors (simplified)
            colors = img_rgb.reshape(-1, 3)
            features['red_ratio'] = np.mean(colors[:, 0]) / 255
            features['green_ratio'] = np.mean(colors[:, 1]) / 255
            features['blue_ratio'] = np.mean(colors[:, 2]) / 255
            
            # 2. Text Density (approximation using edge detection)
            edges = cv2.Canny(img_gray, 100, 200)
            features['text_density'] = np.sum(edges > 0) / edges.size
            
            # 3. Face Detection
            faces = self.face_cascade.detectMultiScale(img_gray, 1.1, 4)
            features['num_faces'] = len(faces)
            features['has_face'] = 1 if len(faces) > 0 else 0
            
            if len(faces) > 0:
                total_face_area = sum([w * h for (x, y, w, h) in faces])
                features['face_area_ratio'] = total_face_area / (img.shape[0] * img.shape[1])
            else:
                features['face_area_ratio'] = 0
            
            # 4. Contrast
            features['contrast'] = img_gray.std()
            
            # 5. Complexity (using edges)
            features['complexity'] = np.sum(edges) / 1000
            
            # 6. Color Temperature (warm vs cool)
            features['warmth'] = (features['red_ratio'] + features['green_ratio']) / 2
            
            return features, img_resized
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None
    
    def prepare_training_data(self, csv_path='data/youtube_dataset.csv'):
        """Prepare dataset for training"""
        df = pd.read_csv(csv_path)
        
        features_list = []
        images_list = []
        labels_list = []
        
        print("Extracting features from thumbnails...")
        
        for idx, row in df.iterrows():
            if pd.isna(row['thumbnail_path']) or not os.path.exists(row['thumbnail_path']):
                continue
            
            features, img = self.extract_thumbnail_features(row['thumbnail_path'])
            
            if features is not None:
                features_list.append(features)
                images_list.append(img / 255.0)  # Normalize
                
                # Create CTR proxy (higher views = higher CTR)
                # Normalize to 0-1 scale
                ctr_proxy = min(row['view_count'] / df['view_count'].quantile(0.95), 1.0)
                labels_list.append(ctr_proxy)
            
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(df)} thumbnails")
        
        # Convert to arrays
        X_features = pd.DataFrame(features_list).values
        X_images = np.array(images_list)
        y = np.array(labels_list)
        
        print(f"\n✅ Prepared {len(y)} samples")
        return X_features, X_images, y
    
    def build_model(self, num_features):
        """Build hybrid CNN + feature model"""
        # Image input branch (CNN)
        img_input = keras.Input(shape=(224, 224, 3), name='image_input')
        
        x = layers.Conv2D(32, 3, activation='relu')(img_input)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.3)(x)
        img_features = layers.Dense(128, activation='relu')(x)
        
        # Engineered features input branch
        feat_input = keras.Input(shape=(num_features,), name='feature_input')
        feat_dense = layers.Dense(64, activation='relu')(feat_input)
        feat_dense = layers.Dropout(0.2)(feat_dense)
        feat_output = layers.Dense(32, activation='relu')(feat_dense)
        
        # Combine both branches
        combined = layers.concatenate([img_features, feat_output])
        z = layers.Dense(64, activation='relu')(combined)
        z = layers.Dropout(0.3)(z)
        output = layers.Dense(1, activation='sigmoid', name='ctr_output')(z)
        
        model = keras.Model(inputs=[img_input, feat_input], outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_features, X_images, y, epochs=20):
        """Train the model"""
        # Split data
        X_feat_train, X_feat_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
            X_features, X_images, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_feat_train_scaled = self.scaler.fit_transform(X_feat_train)
        X_feat_test_scaled = self.scaler.transform(X_feat_test)
        
        # Build model
        self.model = self.build_model(X_features.shape[1])
        
        print("\n🎯 Training model...")
        print(self.model.summary())
        
        # Train
        history = self.model.fit(
            [X_img_train, X_feat_train_scaled],
            y_train,
            validation_data=([X_img_test, X_feat_test_scaled], y_test),
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        
        # Evaluate
        loss, mae = self.model.evaluate([X_img_test, X_feat_test_scaled], y_test)
        print(f"\n✅ Model trained!")
        print(f"   Test MAE: {mae:.4f}")
        
        return history
    
    def save_model(self, path='models/thumbnail_ctr_model'):
        """Save model and scaler"""
        os.makedirs(path, exist_ok=True)
        self.model.save(f'{path}/model.keras')
        with open(f'{path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path='models/thumbnail_ctr_model'):
        """Load saved model"""
        self.model = keras.models.load_model(f'{path}/model.keras')
        with open(f'{path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ Model loaded")
    
    def predict_ctr(self, img_path):
        """Predict CTR for a thumbnail"""
        features, img = self.extract_thumbnail_features(img_path)
        
        if features is None:
            return None
        
        # Prepare inputs
        X_feat = pd.DataFrame([features]).values
        X_feat_scaled = self.scaler.transform(X_feat)
        X_img = np.array([img / 255.0])
        
        # Predict
        ctr_score = self.model.predict([X_img, X_feat_scaled], verbose=0)[0][0]
        
        # Return detailed analysis
        return {
            'ctr_score': float(ctr_score * 100),  # Convert to 0-100 scale
            'features': features,
            'recommendations': self.get_recommendations(features, ctr_score)
        }
    
    def get_recommendations(self, features, ctr_score):
        """Provide improvement suggestions"""
        recommendations = []
        
        if features['has_face'] == 0:
            recommendations.append("Consider adding a face - thumbnails with faces get 38% more clicks")
        
        if features['mean_saturation'] < 80:
            recommendations.append("Increase color saturation for more eye-catching appeal")
        
        if features['contrast'] < 50:
            recommendations.append("Improve contrast to make text more readable")
        
        if features['text_density'] > 0.3:
            recommendations.append("Too much text - simplify for better readability")
        elif features['text_density'] < 0.05:
            recommendations.append("Add clear, bold text to convey the video topic")
        
        if features['warmth'] < 0.4:
            recommendations.append("Add warmer colors (red/orange/yellow) to attract attention")
        
        if not recommendations:
            recommendations.append("Great thumbnail! Minor tweaks could push CTR even higher")
        
        return recommendations


# Training script
if __name__ == "__main__":
    predictor = ThumbnailCTRPredictor()
    
    # Prepare data
    X_features, X_images, y = predictor.prepare_training_data()
    
    # Train model
    history = predictor.train(X_features, X_images, y, epochs=20)
    
    # Save model
    predictor.save_model()
    
    print("\n🎉 Training complete! Model ready for predictions.")