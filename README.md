A full-stack used-car fair price predictor that estimates market value from VIN, vehicle age, and mileage using machine learning. The system exposes a REST API for predictions and a React-based UI for interactive evaluation. Both services are containerized with Docker for reproducibility.
Live Link: https://used-car-price-predictor-kpgukxbxwnpaoz9mgravl9.streamlit.app/
<img width="748" height="1049" alt="image" src="https://github.com/user-attachments/assets/bb7cb071-54cb-481d-bf66-26c4992de425" />

Tech Stack:

    Frontend: React
    
    Backend: FastAPI (Python)
    
    ML: XGBoost
    
    Data: Kaggle Used Car Listings
    
    Infra: Docker, Docker Compose
    
    Evaluation: MAE, R²

Dataset not included due to size (~140MB). See Kaggle link (https://www.kaggle.com/datasets/andreinovikov/used-cars-dataset) for access. Data is used under Kaggle’s public dataset license.
