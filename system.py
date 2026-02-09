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

    def prediction(self, data, seed=None):
        '''
        Method that predicts ammonium nitrate, ammonium chloride, and water content from input.
        
        :param data: n by 7 pandas df with columns [TEMP, RH, NH4+, NA+, SO42-, NO3-, CL-]
        :param seed: optional integer seed that will make predictions reproducible

        Output: n by 4 predicted results
        '''
        processed_subsets = []
        inputs = ['TEMP', 'RH', 'NH4+', 'NA+', 'SO42-', 'NO3-', 'CL-']
        #NH4+ != 0 Case
        mask_nonzero = data['NH4+'] != 0
        if mask_nonzero.any():
            #Scaling water content and chemical input columns for NH4+ != 0 case
            cols_nonzero = ['NA+', 'SO42-', 'NO3-', 'CL-']
            predictors = ['TEMP', 'RH', 'NA+', 'SO42-', 'NO3-', 'CL-']
            df_nonzero = data[mask_nonzero].copy()
            df_nonzero['TEMP_original'] = df_nonzero['TEMP']
            df_nonzero['RH_original'] = df_nonzero['RH']

            raw_inputs = df_nonzero[inputs].copy()

            df_nonzero.loc[:,cols_nonzero] = df_nonzero[cols_nonzero].div(df_nonzero['NH4+'], axis=0)

            #Run phase classifier for NH4+ != 0:
            scalers = self._import_scalers('phase_classifier_nonzero.json')
            df_nonzero['TEMP'] = (df_nonzero['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
            df_nonzero['RH'] = (df_nonzero['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

            phase_classifier_nonzero = self._load_model('phase_classifier_nonzero.keras')

            predictions = phase_classifier_nonzero.predict(df_nonzero[predictors])
            predictions = (predictions >= 0.5).astype(int)
            df_nonzero['phase'] = predictions.flatten()


            #Run models for liquid/mix case of NH4+ != 0
            df_nonzero_liqmix = df_nonzero[df_nonzero['phase']==0].copy()
            if not df_nonzero_liqmix.empty:
                #Ammonium nitrate
                scalers = self._import_scalers('liqmix_amm_nit_nonzero.json')
                df_nonzero_liqmix['TEMP'] = (df_nonzero_liqmix['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
                df_nonzero_liqmix['RH'] = (df_nonzero_liqmix['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

                liqmix_amm_nit_nonzero = self._load_model('liqmix_amm_nit_nonzero.keras')
                predictions = liqmix_amm_nit_nonzero.predict(df_nonzero_liqmix[predictors])
                df_nonzero_liqmix['amm_nit'] = predictions.flatten()

                #Ammonium chloride
                scalers = self._import_scalers('liqmix_amm_chl_nonzero.json')
                df_nonzero_liqmix['TEMP'] = (df_nonzero_liqmix['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
                df_nonzero_liqmix['RH'] = (df_nonzero_liqmix['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

                liqmix_amm_chl_nonzero = self._load_model('liqmix_amm_chl_nonzero.keras')
                predictions = liqmix_amm_chl_nonzero.predict(df_nonzero_liqmix[predictors])
                df_nonzero_liqmix['amm_chl'] = predictions.flatten()

                #water content
                scalers = self._import_scalers('liqmix_water_content_nonzero.json')
                df_nonzero_liqmix['TEMP'] = (df_nonzero_liqmix['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
                df_nonzero_liqmix['RH'] = (df_nonzero_liqmix['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

                liqmix_water_content_nonzero = self._load_model('liqmix_water_content_nonzero.keras')
                predictions = liqmix_water_content_nonzero.predict(df_nonzero_liqmix[predictors])
                df_nonzero_liqmix['water_content'] = predictions.flatten()

                output_cols = ['phase', 'amm_nit', 'amm_chl', 'water_content']
                inputs_for_conversion = raw_inputs.loc[df_nonzero_liqmix.index].copy()
                inputs_for_conversion['TEMP_original'] = df_nonzero_liqmix['TEMP_original']
                inputs_for_conversion['RH_original'] = df_nonzero_liqmix['RH_original']
                df_nonzero_liqmix = self._conversion(inputs_for_conversion, df_nonzero_liqmix[output_cols].copy())

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
            predictors = ['TEMP', 'RH', 'SO42-', 'NO3-', 'CL-']
            df_zero = data[mask_zero].copy()
            df_zero['TEMP_original'] = df_zero['TEMP']
            df_zero['RH_original'] = df_zero['RH']

            raw_inputs = df_zero[inputs].copy()

            df_zero.loc[:,cols_zero] = df_zero[cols_zero].div(df_zero['NA+'], axis=0)

            #Run phase classifier for NH4+ == 0:
            scalers = self._import_scalers('phase_classifier_zero.json')
            df_zero['TEMP'] = (df_zero['TEMP_original'] - scalers['TEMP'][0]) / scalers['TEMP'][1]
            df_zero['RH'] = (df_zero['RH_original'] - scalers['RH'][0]) / scalers['RH'][1]

            phase_classifier_zero = self._load_model('phase_classifier_zero.keras')

            predictions = phase_classifier_zero.predict(df_zero[predictors])
            predictions = (predictions >= 0.5).astype(int)
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
                predictions = liqmix_water_content_zero.predict(df_zero_liqmix[predictors])
                df_zero_liqmix['water_content'] = predictions.flatten()

                output_cols = ['phase', 'amm_nit', 'amm_chl', 'water_content']
                inputs_for_conversion = raw_inputs.loc[df_zero_liqmix.index].copy()
                inputs_for_conversion['TEMP_original'] = df_zero_liqmix['TEMP_original']
                inputs_for_conversion['RH_original'] = df_zero_liqmix['RH_original']
                df_zero_liqmix = self._conversion(inputs_for_conversion, df_zero_liqmix[output_cols].copy())

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
        
        output_cols = ['phase', 'amm_nit', 'amm_chl', 'water_content']
        final_df = pd.concat(processed_subsets).sort_index()
        return final_df[output_cols]              
    
    def evaluate_results(self, actual, predicted, inputs=None):
        '''
        Function that calculates Elvander and Wexler evaluation metrics for system outputs.
        Requires actual values to be known.

        assumes 'actual' column water content is named water_content.
        assumes water content has already been returned to original scale.

        :param actual: actual values for ammonium nitrate, ammonium chloride, and/or water content.
        :param predicted: predicted outputs from MoE system
        :param inputs: optional argument that allows inputs to be used for specialized mass error evaluation metric.

        Output: dictionary of evaluation metrics for each output
        '''
        #initialize metrics
        metrics = {}
        #Loop through columns
        for test_col, pred_col in zip(actual.columns, predicted.columns):
            #Initialize column specific metrics
            metrics[test_col] = {}

            #Extract values
            actual_vals = actual[test_col]
            predicted_vals = predicted[pred_col]

            #If water content is present, calculate mass error
            if test_col == 'water_content' and inputs is not None:
                numerator = abs((actual_vals*18)-(predicted_vals*18))
                denominator = ((inputs['NH4+']*18)+ (inputs['NA+']*23) + (inputs['SO42-']*96) + (inputs['NO3-']*62) + (inputs['CL-']*35.5) + (actual_vals * 18))
                mass_error = (1/len(actual_vals)) * np.sum(numerator/denominator)
                metrics[test_col]['mass_error'] = mass_error
            
            #Calculate mape and add
            mask = actual_vals != 0
            mape = (1/len(actual_vals[mask]))*np.sum(abs(np.array(actual_vals[mask])-np.array(predicted_vals[mask]))/(np.array(actual_vals[mask])))
            metrics[test_col]['mape'] = mape

            #calculate NMAE and add
            nmae = (1/np.mean(actual[test_col]))*(1/len(actual[test_col]))*np.sum(abs(np.array(actual[test_col])-np.array(predicted[pred_col])))
            metrics[test_col]['nmae'] = nmae

            #Calculate rmse and add
            rmse = np.sqrt((1/len(actual[test_col]))*np.sum((np.array(actual[test_col])-np.array(predicted[pred_col]))**2))
            metrics[test_col]['rmse'] = rmse

        return metrics
    
    def _conversion(self, inputs, outputs):
        '''
        Internal function that converts outputs from scaled space to original space.
        Multiplies partial pressure products by solid partial pressure product at equivalent temperature,
        undoes water content scaling based on input compound.

        :param inputs: dataframe of raw, unscaled inputs [TEMP_original, RH_original, NH4+, NA+, SO42-, NO3-, CL-]
        :param outputs: dataframe of scaled outputs [phase, amm_nit, amm_chl, water_content]

        Output: unscaled outputs
        '''
        cols = outputs.columns
        #Ammonium Nitrate
        if 'amm_nit' in cols:
            solid_amm_nit_nonzero = self._load_model('solid_amm_nit_nonzero.pkl', keras=False)
            b = solid_amm_nit_nonzero.coef_[0]
            a = np.exp(solid_amm_nit_nonzero.intercept_)
            outputs.loc[:,'amm_nit'] = outputs['amm_nit']*(a * np.exp(b * (1/inputs['TEMP_original'])))

        #Ammonium Chloride
        if 'amm_chl' in cols:
            solid_amm_chl_nonzero = self._load_model('solid_amm_chl_nonzero.pkl', keras=False)
            b = solid_amm_chl_nonzero.coef_[0]
            a = np.exp(solid_amm_chl_nonzero.intercept_)
            outputs.loc[:,'amm_chl'] = outputs['amm_chl']*(a * np.exp(b * (1/inputs['TEMP_original'])))

        #Water content
        if 'water_content' in cols:
            rh_factor = (inputs['RH_original'])/(1-inputs['RH_original'])

            #NH4+ != 0
            nz_mask = inputs['NH4+'] != 0
            if nz_mask.any():
                ion_sum_nz = (inputs.loc[nz_mask, ['NH4+', 'NA+', 'SO42-', 'NO3-', 'CL-']].sum(axis=1))
                outputs.loc[nz_mask, 'water_content'] *= (ion_sum_nz * rh_factor.loc[nz_mask])
            
            #NH4+ = 0
            z_mask = inputs['NH4+'] == 0
            if z_mask.any():
                ion_sum_z = (inputs.loc[z_mask, ['NA+', 'SO42-', 'NO3-', 'CL-']].sum(axis=1))
                outputs.loc[z_mask, 'water_content'] *= (ion_sum_z * rh_factor.loc[z_mask])
        
        return outputs
    
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
        Helper function to import keras/pickled models from appropriate filepath.
        
        :param model_name: name of model
        :param keras: boolean indicating if model is .keras or not

        Output: model object
        '''
        #get current path
        current_dir = Path(__file__).parent
        #Construct filepath
        file_path = current_dir / 'models' / model_name
        #Load and return model
        if keras:
            model = tf.keras.models.load_model(file_path, compile=False)
            return model
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model