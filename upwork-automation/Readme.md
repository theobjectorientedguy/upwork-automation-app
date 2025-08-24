To install dependencies: 
pip install -r requirements.txt

To run the application via Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload