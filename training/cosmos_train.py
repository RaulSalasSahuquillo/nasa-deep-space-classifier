import os
import pandas as pd
from sklearn.model_selection import train_test_split

class_to_idx = {
    'star': 0,
    'galaxy': 1,
    'quasar': 2,
    'nebula': 3,
    'planet': 4
}

data = []

sources = [
    ('/home/rsalas/Documentos/NASA_TRAINING/training/data/stars', 'star'),
    ('/home/rsalas/Documentos/NASA_TRAINING/training/data/galaxies', 'galaxy'),
    ('/home/rsalas/Documentos/NASA_TRAINING/training/data/quasars', 'quasar'),
    ('/home/rsalas/Documentos/NASA_TRAINING/training/data/nebulae', 'nebula'),
    ('/home/rsalas/Documentos/NASA_TRAINING/training/data/planets', 'planet'),
]

for folder_path, class_name in sources:
    if os.path.exists(folder_path):
        label = class_to_idx[class_name]
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in files:
            img_path = os.path.join(folder_path, img_name)
            data.append({'image_path': img_path, 'label': label})

if not data:
    raise ValueError(
        "No se encontraron imágenes en training/data/. "
        "Asegúrate de haber puesto imágenes (.jpg/.png) en las subcarpetas de training/data/."
    )

df = pd.DataFrame(data)

# Split 80% entrenamiento / 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Save CSV files for main.ipynb / main.py
train_df.to_csv('cosmos_train.csv', index=False)
test_df.to_csv('cosmos_test.csv', index=False)

print("¡Archivos cosmos_train.csv y cosmos_test.csv generados con éxito!")