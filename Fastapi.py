from House_model import House_model
from fastapi import FastAPI
import uvicorn
import pickle

app = FastAPI(title = 'Model for House Price Prediction', 
            description = 'Accurately predicting prices of houses in Nigeria!')

@app.get("/")
def home():
    return "Welcome! API is working perfectly well. Use /docs to proceed to make your predictions."

@app.post("/predict")
def make_prediction(house: House_model):
    year = house.date.year
    month = house.date.month
    day = house.date.day
    bedrooms = house.bedrooms
    bathrooms = house.bathrooms
    sqft_living = house.sqft_living
    sqft_lot = house.sqft_lot
    floors = house.floors
    waterfront = house.waterfront
    view = house.view
    condition = house.condition
    grade = house.grade
    sqft_above = house.sqft_above
    sqft_basement = house.sqft_basement
    yr_built = house.yr_built
    yr_renovated = house.yr_renovated
    zipcode = house.zipcode
    lat = house.lat
    long = house.long
    sqft_living15 = house.sqft_living15
    sqft_lot15 = house.sqft_lot15

    data = [[year, month, day, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
     condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15 ]]
    
    loaded_model = pickle.load(open('House_Prediction.pkl', 'rb'))
    
    prediction = loaded_model.predict(data)

    return {"House price prediction": prediction.tolist()}

#run
if __name__ == '__main__':
    uvicorn.run(app)
