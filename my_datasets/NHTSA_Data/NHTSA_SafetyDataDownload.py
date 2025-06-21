#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:20:41 2023
@author: habbas3
"""

import requests
import time
import pandas as pd

BASE_URL = 'https://api.nhtsa.gov/SafetyRatings'

def get_data(endpoint, retries=3, timeout=120):
    for _ in range(retries):
        try:
            with requests.Session() as s:
                response = s.get(f"{BASE_URL}/{endpoint}", timeout=timeout, verify=False)
            
            if response.status_code == 200:
                return response.json()['Results']
            else:
                print(f"Error {response.status_code} for endpoint {endpoint}: {response.text}")
        except requests.ConnectionError:
            print("Connection error. Retrying...")
            time.sleep(10)
    
    print(f"Failed to fetch data for endpoint {endpoint} after {retries} attempts.")
    return []

# 1. Get all model years
model_years = get_data('')
all_vehicle_data = []

for year_data in model_years:
    year = year_data['ModelYear']
    
    # 2. For each model year, get all makes
    makes = get_data(f'modelyear/{year}/')
    
    for make_data in makes:
        make_name = make_data['Make']
        
        # 3. For each make, get all models
        models = get_data(f'modelyear/{year}/make/{make_name}/')
        
        for model_data in models:
            model_name = model_data['Model']
            
            # 4. For each model, get all vehicle IDs
            vehicles = get_data(f'modelyear/{year}/make/{make_name}/model/{model_name}')
            
            for vehicle in vehicles:
                vehicle_id = vehicle['VehicleId']
                
                # 5. Fetch data for each vehicle ID
                vehicle_data = get_data(f'VehicleId/{vehicle_id}')
                all_vehicle_data.extend(vehicle_data)
                
                # Sleep to prevent hitting rate limits (you can adjust this)
                time.sleep(3)

df = pd.DataFrame(all_vehicle_data)
df.to_excel('nhtsa_vehicle_SafetyRatingsData.xlsx', index=False, engine='openpyxl')
