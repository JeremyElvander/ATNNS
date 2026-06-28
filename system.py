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

class ATNNS():
    def __init__(self, windows=False):
        self.windows = windows

    def prediction(self, data):
        '''
        Method that predicts ammonium nitrate, ammonium chloride, and water content from input.
        
        :param data: n by 7 pandas df with columns [TEMP, RH, NH4+, NA+, SO42-, NO3-, CL-]

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
        outputs = outputs.astype('float64')
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
            if not self.windows:
                model = tf.keras.models.load_model(file_path, compile=False)
                return model
            import zipfile, json, tempfile, os, io, re
            import h5py

            def _strip_batch_input_shape(obj):
                # Keras 3 does not accept batch_input_shape on Dense layers (Keras 2 artifact)
                if isinstance(obj, dict):
                    if obj.get('class_name') != 'InputLayer':
                        inner = obj.get('config')
                        if isinstance(inner, dict):
                            inner.pop('batch_input_shape', None)
                    for v in obj.values():
                        _strip_batch_input_shape(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _strip_batch_input_shape(item)

            def _fix_h5_paths(weights_bytes):
                # tensorflow-macos numbers H5 layer groups as dense, dense_2, dense_4...
                # Keras 3 on Windows expects dense, dense_1, dense_2...
                # This renames them sequentially without modifying files on disk.
                with h5py.File(io.BytesIO(weights_bytes), 'r') as orig:
                    if '_layer_checkpoint_dependencies' not in orig:
                        return weights_bytes
                    orig_lcp = orig['_layer_checkpoint_dependencies']
                    groups = sorted(orig_lcp.keys(), key=lambda n: int(re.search(r'_(\d+)$', n).group(1)) if re.search(r'_(\d+)$', n) else 0)
                    new_buf = io.BytesIO()
                    with h5py.File(new_buf, 'w') as new:
                        for key in orig.keys():
                            if key != '_layer_checkpoint_dependencies':
                                orig.copy(key, new)
                        new_lcp = new.create_group('_layer_checkpoint_dependencies')
                        for i, old_name in enumerate(groups):
                            new_name = 'dense' if i == 0 else f'dense_{i}'
                            orig_lcp.copy(old_name, new_lcp, name=new_name)
                    return new_buf.getvalue()

            import keras as _keras
            keras_major = int(_keras.__version__.split('.')[0])
            with zipfile.ZipFile(file_path, 'r') as zf:
                config = json.loads(zf.read('config.json'))
                _strip_batch_input_shape(config)
                weights = zf.read('model.weights.h5')
                if keras_major >= 3:
                    weights = _fix_h5_paths(weights)
                fd, tmp_path = tempfile.mkstemp(suffix='.keras')
                os.close(fd)
                try:
                    with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as out_zf:
                        out_zf.writestr('config.json', json.dumps(config))
                        out_zf.writestr('model.weights.h5', weights)
                        for name in zf.namelist():
                            if name not in ('config.json', 'model.weights.h5'):
                                out_zf.writestr(name, zf.read(name))
                    model = tf.keras.models.load_model(tmp_path, compile=False)
                finally:
                    os.unlink(tmp_path)
            return model
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model


class ATNNS_GPU(ATNNS):
    def __init__(self, windows=False):
        '''
        '''
        super().__init__(windows=windows)
        
        #Retrieve keras networks
        self.models = {
            'phase_classifier_nonzero': self._load_model('phase_classifier_nonzero.keras'),
            'phase_classifier_zero': self._load_model('phase_classifier_zero.keras'),
            'liqmix_amm_chl_nonzero': self._load_model('liqmix_amm_chl_nonzero.keras'),
            'liqmix_amm_nit_nonzero': self._load_model('liqmix_amm_nit_nonzero.keras'),
            'liqmix_water_content_nonzero': self._load_model('liqmix_water_content_nonzero.keras'),
            'liqmix_water_content_zero': self._load_model('liqmix_water_content_zero.keras'),
        }

        #Retrieve and save solid regression constants to Tensors
        #Ammonium Nitrate
        solid_amm_nit_nonzero = self._load_model('solid_amm_nit_nonzero.pkl', keras=False)
        self.solid_nit_a = tf.constant(np.exp(solid_amm_nit_nonzero.intercept_), dtype=tf.float32)
        self.solid_nit_b = tf.constant(solid_amm_nit_nonzero.coef_[0], dtype=tf.float32)
        
        #Ammonium Chloride
        solid_amm_chl_nonzero = self._load_model('solid_amm_chl_nonzero.pkl', keras=False)
        self.solid_chl_a = tf.constant(np.exp(solid_amm_chl_nonzero.intercept_), dtype=tf.float32)
        self.solid_chl_b = tf.constant(solid_amm_chl_nonzero.coef_[0], dtype=tf.float32)
        
        #Preloading scalers
        scaler_files = [
            'liqmix_amm_chl_nonzero',
            'liqmix_amm_nit_nonzero',
            'liqmix_water_content_nonzero',
            'liqmix_water_content_zero',
            'phase_classifier_nonzero',
            'phase_classifier_zero'
        ]

        self.scalers = {}
        for scaler in scaler_files:
            filename = scaler + '.json'
            s = self._import_scalers(filename)

            #Store mean and sd as tensors for fast math
            self.scalers[scaler] = {
                #[mean, std] tensor format for TEMP and RH
                'TEMP_mean': tf.constant(s['TEMP'][0], dtype=tf.float32, shape=()),
                'TEMP_std': tf.constant(s['TEMP'][1], dtype=tf.float32, shape=()),
                'RH_mean': tf.constant(s['RH'][0], dtype=tf.float32, shape=()),
                'RH_std': tf.constant(s['RH'][1], dtype=tf.float32, shape=()),
            }
    
    #Decorator to compile logic into GPU graph
    @tf.function
    def gpu_prediction(self, input_tensor):
        '''
        '''

        def scale_temp_rh(t, r, key):
            '''
            Helper function to apply temperature and RH scaling inside the graph structure.
            '''
            t_scaled = (t-self.scalers[key]['TEMP_mean']) / self.scalers[key]['TEMP_std']
            r_scaled = (r-self.scalers[key]['RH_mean']) / self.scalers[key]['RH_std']

            return t_scaled, r_scaled

        #Extract necessary individual columns
        nh4 = input_tensor[:, 2:3]
        na = input_tensor[:, 3:4]
        temp = input_tensor[:, 0:1]
        rh = input_tensor[:, 1:2]

        #Creating mask for NH4+ gate
        nh4_present = nh4>0

        #Set up scaling denominators, avoid dividing by zero by replacing NH4+ = 0 with 1 temporarily
        safe_nh4 = tf.where(nh4==0, tf.ones_like(nh4), nh4)
        safe_na = tf.where(na==0, tf.ones_like(na), na)

        #Perform scaling on all chemical inputs
        #NH4+ != 0 case, avoids NaN without partitioning data
        chem_nz = input_tensor[:, 3:7] / safe_nh4
        #NH4+ = 0 case for all inputs
        chem_z = input_tensor[:, 4:7] / safe_na
        

        # NH4+ != 0 CASE
        #Phase classification
        t_phase_nz, r_phase_nz = scale_temp_rh(temp, rh, 'phase_classifier_nonzero')
        phase_inputs_nz = tf.concat([t_phase_nz, r_phase_nz, chem_nz], axis=1)
        phase_nz = tf.cast(self.models['phase_classifier_nonzero'](phase_inputs_nz) >= 0.5, tf.float32)

        #Liquid/Mix amm_nit, amm_chl, and water content
        t_lm_nit, r_lm_nit = scale_temp_rh(temp, rh, 'liqmix_amm_nit_nonzero')
        liq_nit_nz = self.models['liqmix_amm_nit_nonzero'](tf.concat([t_lm_nit, r_lm_nit, chem_nz], axis=1))

        t_lm_chl, r_lm_chl = scale_temp_rh(temp, rh, 'liqmix_amm_chl_nonzero')
        liq_chl_nz = self.models['liqmix_amm_chl_nonzero'](tf.concat([t_lm_chl, r_lm_chl, chem_nz], axis=1))

        t_lm_wc, r_lm_wc = scale_temp_rh(temp, rh, 'liqmix_water_content_nonzero')
        liq_wc_nz = self.models['liqmix_water_content_nonzero'](tf.concat([t_lm_wc, r_lm_wc, chem_nz], axis=1))

        #Solid case amm_nit, amm_chl, and water content
        sol_nit_nz = self.solid_nit_a * tf.math.exp(self.solid_nit_b * (1 / temp))

        sol_chl_nz = self.solid_chl_a * tf.math.exp(self.solid_chl_b * (1 / temp))

        #0 for water content solid case
        sol_wc_nz = tf.zeros_like(liq_wc_nz)

        #Merging NH4+ != 0 liquid/mix and solid cases based on phase mask (0 = liquid/mix, 1 = solid)
        #Creating mask, multiplying models by 0/1 to 'activate' correct results
        is_solid_nz = tf.cast(phase_nz > 0, tf.float32)
        result_nit_nz = (1 - is_solid_nz) * liq_nit_nz + (is_solid_nz * sol_nit_nz)
        result_chl_nz = (1 - is_solid_nz) * liq_chl_nz + (is_solid_nz * sol_chl_nz)
        result_wc_nz = (1 - is_solid_nz) * liq_wc_nz + (is_solid_nz * sol_wc_nz)


        # NH4 == 0 CASE
        #Phase classification
        t_phase_z, r_phase_z = scale_temp_rh(temp, rh, 'phase_classifier_zero')
        phase_inputs_z = tf.concat([t_phase_z, r_phase_z, chem_z], axis=1)
        phase_z = tf.cast(self.models['phase_classifier_zero'](phase_inputs_z) >= 0.5, tf.float32)

        #Liquid/mix water content prediction
        t_lm_wc_z, r_lm_wc_z = scale_temp_rh(temp, rh, 'liqmix_water_content_zero')
        liq_wc_z = self.models['liqmix_water_content_zero'](tf.concat([t_lm_wc_z, r_lm_wc_z, chem_z], axis=1))

        #Merge NH4+ = 0 liquid/mix and solid cases (0 for solid case for water content), (for phase, 0 = liquid/mix, 1 = solid)
        is_solid_z = tf.cast(phase_z > 0, tf.float32)
        #Solid water content case is 0
        result_wc_z = (1 - is_solid_z) * liq_wc_z

        #Amm nit and amm chl results are 0
        result_nit_z = tf.zeros_like(result_wc_z)
        result_chl_z = tf.zeros_like(result_wc_z)


        # MERGE ALL RESULTS AND UNDO SCALING
        #Using original NH4+ mask to select correct outputs
        mask = tf.cast(nh4_present, tf.float32)

        final_phase = mask * phase_nz + (1 - mask) * phase_z
        final_nit = mask * result_nit_nz + (1 - mask) * result_nit_z
        final_chl = mask * result_chl_nz + (1 - mask) * result_chl_z
        final_wc = mask * result_wc_nz + (1 - mask) * result_wc_z

        scaled_outputs = tf.concat([final_phase, final_nit, final_chl, final_wc], axis=1)

        #Convert scaled outputs to original values with original raw inputs
        return self._gpu_conversion(input_tensor, scaled_outputs)
        
    
    def _gpu_conversion(self, inputs, outputs):
        '''
        '''
        temp = inputs[:, 0:1]
        rh = inputs[:, 1:2]
        nh4 = inputs[:, 2:3]

        phase = outputs[:, 0:1]
        amm_nit = outputs[:, 1:2]
        amm_chl = outputs[:, 2:3]
        water_content = outputs[:, 3:4]

        #Unscaling amm_nit and amm_chl based on solid case
        sol_factor_nit = self.solid_nit_a * tf.math.exp(self.solid_nit_b * (1 / temp))
        unscaled_nit = tf.where(phase>0, amm_nit, amm_nit * sol_factor_nit)

        sol_factor_chl = self.solid_chl_a * tf.math.exp(self.solid_chl_b * (1 / temp))
        unscaled_chl = tf.where(phase>0, amm_chl, amm_chl * sol_factor_chl)

        #Unscaling water content for both cases
        rh_factor = rh / (1.0 - rh)

        #Calculating ion sums for both conditions
        ion_sum_nz = tf.reduce_sum(inputs[:, 2:7], axis=1, keepdims=True)
        ion_sum_z = tf.reduce_sum(inputs[:, 3:7], axis=1, keepdims=True)

        #Use mask to choose correct ion sum for each output
        ion_sum = tf.where(nh4>0, ion_sum_nz, ion_sum_z)
        unscaled_water_content = water_content * ion_sum * rh_factor

        #Return Nx4 vector
        return tf.concat([phase, unscaled_nit, unscaled_chl, unscaled_water_content], axis=1)

        
