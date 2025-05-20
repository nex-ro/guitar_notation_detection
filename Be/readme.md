# FastAPI Project

## Cara Menjalankan
requirement  :
python 3 , git , npm

Buat virtual environment (di paling luar):
python -m venv venv
source venv/Scripts/activate

pip install FastAPI
pip install "uvicorn[standard]"
pip install tensorflow
pip install librosa
pip install scikit-image
pip install python-multipart

run :
$ python -m uvicorn app:app --reload