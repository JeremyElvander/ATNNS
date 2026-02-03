import models
import temp_rh_scalers

import numpy as np
import pandas as pd
import keras
import tensorflow as tf

from pathlib import Path
import json
import pickle
from concurrent.futures import ProcessPoolExecutor

class NeuralSystem():
    def __init__(self, name, **kwargs):
        self.name = name
    
    def prediction(self, data, phase=None, actual=None, seed=None):
        '''
        Method that predicts ammonium nitrate, ammonium chloride, and water content from input.
        
        :param data: n by 7 pandas df with columns [TEMP, RH, NH4+, NA+, SO42-, NO3-, CL-]
        :param phase: n by 1 pandas df with column 'phase', binary indicator of phase, 0 = liquid/mix; 1 = solid.
            When provided, calculates evaluation metrics from phase predictions.
        :param actual: optional n by 3 pandas df with columns [amm_nit, amm_chl, water_content].
            When provided, calculates evaluation metrics from predictions.
        :param seed: optional integer seed that will make predictions reproducible

        Output: n by 3
        '''
        processed_subsets = []

        #NH4+ != 0 Case
        mask_nonzero = data['NH4+'] != 0
        if mask_nonzero.any():
            #Scaling water content and chemical input columns for NH4+ != 0 case
            cols_nonzero = ['NA+', 'SO42-', 'NO3-', 'CL-']
            df_nonzero = data[mask_nonzero].copy()
            df_nonzero['TEMP_original'] = df_nonzero['TEMP']
            df_nonzero['RH_original'] = df_nonzero['RH']
            df_nonzero.loc[:,cols_nonzero] = df_nonzero[cols_nonzero].div(df_nonzero['NH4+'], axis=0)

            #Run phase classifier for NH4+ != 0:
            scalers = self._import_scalers('phase_classifier_nonzero.json')
            df_nonzero['TEMP'] = (df_nonzero['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
            df_nonzero['RH'] = (df_nonzero['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

            phase_classifier_nonzero = self._load_model('phase_classifier_nonzero.keras')
            predictions = phase_classifier_nonzero.predict(df_nonzero.drop(['NH4+', 'TEMP_original', 'RH_original'], axis=1))
            df_nonzero['phase'] = predictions.flatten()


            #Run models for liquid/mix case of NH4+ != 0
            df_nonzero_liqmix = df_nonzero[df_nonzero['phase']==0].copy()
            if not df_nonzero_liqmix.empty:
                #Ammonium nitrate
                scalers = self._import_scalers('liqmix_amm_nit_nonzero.json')
                df_nonzero_liqmix['TEMP'] = (df_nonzero_liqmix['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
                df_nonzero_liqmix['RH'] = (df_nonzero_liqmix['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

                liqmix_amm_nit_nonzero = self._load_model('liqmix_amm_nit_nonzero.keras')
                predictions = liqmix_amm_nit_nonzero.predict(df_nonzero_liqmix.drop(['NH4+', 'TEMP_original', 'RH_original', 'phase'], axis=1))
                df_nonzero_liqmix['amm_nit'] = predictions.flatten()

                #Ammonium chloride
                scalers = self._import_scalers('liqmix_amm_chl_nonzero.json')
                df_nonzero_liqmix['TEMP'] = (df_nonzero_liqmix['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
                df_nonzero_liqmix['RH'] = (df_nonzero_liqmix['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

                liqmix_amm_chl_nonzero = self._load_model('liqmix_amm_chl_nonzero.keras')
                predictions = liqmix_amm_chl_nonzero.predict(df_nonzero_liqmix.drop(['NH4+', 'TEMP_original', 'RH_original', 'phase', 'amm_nit'], axis=1))
                df_nonzero_liqmix['amm_chl'] = predictions.flatten()

                #water content
                scalers = self._import_scalers('liqmix_water_content_nonzero.json')
                df_nonzero_liqmix['TEMP'] = (df_nonzero_liqmix['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
                df_nonzero_liqmix['RH'] = (df_nonzero_liqmix['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

                liqmix_water_content_nonzero = self._load_model('liqmix_water_content_nonzero.keras')
                predictions = liqmix_water_content_nonzero.predict(df_nonzero_liqmix.drop(['NH4+', 'TEMP_original', 'RH_original', 'phase', 'amm_nit', 'amm_chl'], axis=1))
                df_nonzero_liqmix['water_content'] = predictions.flatten()

                processed_subsets.append(df_nonzero_liqmix)


            #Run models for solid case of NH4+ != 0
            df_nonzero_solid = df_nonzero[df_nonzero['phase'] != 0].copy()
            if not df_nonzero_solid.empty:
                #Ammonium nitrate
                solid_amm_nit_nonzero = self._load_model('solid_amm_nit_nonzero.pkl', keras=False)
                b = solid_amm_nit_nonzero.coef_[0]
                a = np.exp(solid_amm_nit_nonzero.intercept_)
                df_nonzero_solid['amm_nit'] = (a * np.exp(b * (1/df_nonzero_solid['TEMP_original'])))

                #Ammonium chloride
                solid_amm_chl_nonzero = self._load_model('solid_amm_chl_nonzero.pkl', keras=False)
                b = solid_amm_chl_nonzero.coef_[0]
                a = np.exp(solid_amm_chl_nonzero.intercept_)
                df_nonzero_solid['amm_chl'] = (a * np.exp(b * (1/df_nonzero_solid['TEMP_original'])))

                #Water content
                df_nonzero_solid['water_content'] = 0

                processed_subsets.append(df_nonzero_solid)


        #NH4+ = 0 Case
        mask_zero = data['NH4+'] == 0
        if mask_zero.any():
            #Scaling water content and chemical input columns for NH4+ == 0 case
            cols_zero = ['SO42-', 'NO3-', 'CL-']
            df_zero = data[mask_zero].copy()
            df_zero['TEMP_original'] = df_zero['TEMP']
            df_zero['RH_original'] = df_zero['RH']
            #df_zero['water_content'] = df_zero['water_content']/((df_zero['NA+']+df_zero['SO42-']+df_zero['NO3-']+df_zero['CL-']) * ((df_zero['RH'])/(1-df_zero['RH'])))
            df_zero.loc[:,cols_zero] = df_zero[cols_zero].div(df_zero['NA+'], axis=0)
            

            #Run phase classifier for NH4+ == 0:
            scalers = self._import_scalers('phase_classifier_zero.json')
            df_zero['TEMP'] = (df_zero['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
            df_zero['RH'] = (df_zero['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

            phase_classifier_zero = self._load_model('phase_classifier_zero.keras')
            predictions = phase_classifier_zero.predict(df_zero.drop(['NH4+', 'NA+', 'TEMP_original', 'RH_original'], axis=1))
            df_zero['phase'] = predictions.flatten()



            #Run model for liquid/mix case for NH4+ == 0
            df_zero_liqmix = df_zero[df_zero['phase']==0].copy()

            if not df_zero_liqmix.empty:
                #Ammonium nitrate
                df_zero_liqmix['amm_nit'] = 0

                #Ammonium chloride
                df_zero_liqmix['amm_chl'] = 0

                #water content
                scalers = self._import_scalers('liqmix_water_content_zero.json')
                df_zero_liqmix['TEMP'] = (df_zero_liqmix['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
                df_zero_liqmix['RH'] = (df_zero_liqmix['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

                liqmix_water_content_zero = self._load_model('liqmix_water_content_zero.keras')
                predictions = liqmix_water_content_zero.predict(df_zero_liqmix.drop(['NH4+', 'NA+', 'TEMP_original', 'RH_original', 'phase', 'amm_nit', 'amm_chl'], axis=1))
                df_zero_liqmix['water_content'] = predictions.flatten()

                processed_subsets.append(df_zero_liqmix)


            #Run information for solid case for NH4+ == 0
            df_zero_solid = df_zero[df_zero['phase']!=0].copy()

            if not df_zero_solid.empty:
                #Ammonium nitrate
                df_zero_solid['amm_nit'] = 0

                #Ammonium chloride
                df_zero_solid['amm_chl'] = 0

                #Water content
                df_zero_solid['water_content'] = 0

                processed_subsets.append(df_zero_solid)
        
        if not processed_subsets:
            return pd.DataFrame()
        
        final_df = pd.concat(processed_subsets).sort_index()
        return final_df[['amm_nit', 'amm_chl', 'water_content']]
     
    
    def _evaluation_metrics():
        raise NotImplementedError
    
    def _import_scalers(self, file_name):
        '''
        Helper function to import temperature/RH scalers from appropriate json file.
        
        :param file_name: name of scaler file.

        Output scalers: dictionary containing (mean, std) for temperature and relative humidity
        '''
        #Get current path
        current_dir = Path(__file__).parent
        #Construct filepath
        file_path = current_dir / 'temp_rh_scalers' / file_name
        #Load json file
        with open(file_path, 'r') as file:
            scalers = json.load(file)
            return scalers
    
    def _load_model(self, model_name, keras=True):
        '''
        Docstring for _load_model
        
        :param model_name: Description
        '''
        #get current path
        current_dir = Path(__file__).parent
        #Construct filepath
        file_path = current_dir / 'models' / model_name
        #Load and return model
        if keras:
            model = tf.keras.models.load_model(file_path)
            return model
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model